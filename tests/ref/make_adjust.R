# Reference values for the adjustments (normalise*) and normaliseBlocked.
source("tests/ref/_common.R")
suppressPackageStartupMessages(library(SingleCellExperiment))
ns <- function(n) get(n, envir = asNamespace("SpaNorm"))

spe <- zellkonverter::readH5AD("tests/ref/data/dlpfc_small.h5ad")
# small slice (every 3rd spot, 40 genes spread over the 300) keeps the committed reference file small
cells <- seq(1, ncol(spe), by = 3)
Y <- as.matrix(assay(spe, "counts"))[round(seq(1, 300, length.out = 40)), cells]
dimnames(Y) <- NULL
xy <- reducedDim(spe, "spatial")[cells, ]
nc <- ncol(Y)
LS <- colSums(Y) / mean(colSums(Y))

fit_it <- function(batch, seed) {
  set.seed(seed)
  ns("fitSpaNorm")(Y = Y, coords = xy, sample.p = 0.25, gene.model = "nb", df.tps = 2,
                   lambda.a = 1e-4, batch = batch, LS = LS, msgfun = function(...) NULL,
                   tol = 1e-2, maxit.nb = 10, maxit.psi = 3, backend = "cpu")
}
pack_fit <- function(f) {
  list(ngenes = f@ngenes, ncells = f@ncells, gene_model = f@gene.model, df_tps = f@df.tps,
       sample_p = f@sample.p, lambda_a = f@lambda.a, W = f@W, alpha = f@alpha,
       gmean = f@gmean, psi = f@psi, wtype = as.character(f@wtype), loglik = f@loglik,
       sampling = as.character(f@sampling))
}
methods <- list(logpac = ns("normaliseLogPAC"), pearson = ns("normalisePearson"),
                medbio = ns("normaliseMedianBio"), meanbio = ns("normaliseMeanBio"),
                meanls = ns("normaliseMeanLS"))
run_all <- function(fit, extra = list()) {
  out <- list()
  for (sf in c(1, 2.5)) {
    for (m in names(c(methods, extra))) {
      f <- c(methods, extra)[[m]]
      out[[sprintf("%s_sf%s", m, sub("\\.", "p", format(sf)))]] <- unname(f(Y, sf, fit))
    }
  }
  out
}

out <- list(Y = Y)
f1 <- fit_it(NULL, 1)
out$nobatch <- pack_fit(f1)
out$nobatch$adj <- run_all(f1)

batch <- factor(ifelse(xy[, 1] > median(xy[, 1]), "b2", "b1"))
f2 <- fit_it(batch, 2)
out$batch <- pack_fit(f2)
out$batch$batch <- as.character(batch)
out$batch$adj <- run_all(f2, list(meanbatch = ns("normaliseMeanBatch")))

# R's own gene-blocked result (block.size in elements) for logpac and pearson
blk <- ns("normaliseBlocked")
out$blocked <- list(
  logpac = unname(blk(ns("normaliseLogPAC"), Y, 1, f1, block.size = 8e3)),
  pearson = unname(blk(ns("normalisePearson"), Y, 2.5, f1, block.size = 8e3)),
  nblocks = as.integer(ceiling(nrow(Y) * ncol(Y) / 8e3))
)
# degenerate dispersions on the same (no-batch) fit
degen <- function(psi) {
  f <- f1; f@psi <- psi
  c(list(psi = psi), lapply(methods[c("logpac", "pearson", "medbio", "meanbio")], function(g) unname(g(Y, 2.5, f))))
}
psiA <- f1@psi; psiA[1] <- 0
psiB <- f1@psi; psiB[1] <- 0; psiB[2] <- NaN; psiB[3] <- 1e-300; psiB[4] <- 1e6
out$degen_zero <- degen(psiA)
out$degen_mixed <- degen(psiB)
write_ref(out, "adjust")
