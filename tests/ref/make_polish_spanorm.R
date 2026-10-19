# Reference values for polishSpaNorm (R/polishSpaNorm.R, polishSpaNormModel.R,
# polishSpaNormJoint.R) on small real SpaNorm fits (helper-polish.R fixtures, built
# from the dlpfc_small counts so the SpaNorm fit can be injected into Python).
#
# Two fixtures: "a" (no batch; SpaNormSVG run first so the null fit exists) and "b"
# (two-level batch, one gene with no counts in level s2). For each polish run we export
# every stored fit (SpaNorm, SpaNormUnpolished, SpaNormNull, SpaNormNullUnpolished) and
# the rewritten logcounts. Python seeds spanorm / spanorm_null from the *input* arrays
# (spanorm_svg arrives in a later task) and compares everything it stores.
source("tests/ref/_common.R")
suppressPackageStartupMessages({
  library(SpatialExperiment)
  library(SingleCellExperiment)
})

sce <- zellkonverter::readH5AD("tests/ref/data/dlpfc_small.h5ad")
Yall <- as.matrix(assay(sce, "counts"))
xyall <- as.matrix(reducedDim(sce, "spatial"))

NG <- 40L
NC <- 500L
top <- order(-rowSums(Yall))[seq_len(NG)]
Y <- Yall[top, seq_len(NC)]
xy <- xyall[seq_len(NC), , drop = FALSE]
storage.mode(Y) <- "integer"
colnames(Y) <- paste0("c", seq_len(NC))

mk_spe <- function(Y, xy) {
  spe <- SpatialExperiment(assays = list(counts = Y), spatialCoords = xy)
  colnames(spe) <- paste0("c", seq_len(ncol(Y)))
  rownames(spe) <- rownames(Y)
  spe
}

pack_fit <- function(f) {
  out <- list(
    ngenes = f@ngenes, ncells = f@ncells, df_tps = f@df.tps, lambda_a = f@lambda.a,
    W = f@W, W_colnames = colnames(f@W), alpha = f@alpha, gmean = f@gmean, psi = f@psi,
    wtype = as.character(f@wtype), loglik = f@loglik, sampling = as.character(f@sampling),
    polished = isPolished(f)
  )
  if (length(f@batch)) out$batch <- if (is.matrix(f@batch)) unname(f@batch) else as.character(f@batch)
  if (isPolished(f)) {
    pol <- SpaNorm:::.polishSlot(f)
    s <- pol$settings
    s <- s[!names(s) %in% "SpaNorm"]
    names(s) <- gsub(".", "_", names(s), fixed = TRUE)
    s <- lapply(s, function(v) if (is.logical(v)) as.integer(v) else v)
    out$settings <- s
    g <- pol$genes
    gl <- lapply(as.list(g), function(v) {
      if (is.logical(v)) as.integer(v) else if (is.character(v)) { v[is.na(v)] <- ""; v } else v
    })
    out$genes <- gl
  }
  out
}

pack_run <- function(spe, warns, msgs) {
  md <- S4Vectors::metadata(spe)
  out <- list(logcounts = unname(as.matrix(assay(spe, "logcounts"))),
              warns = if (length(warns)) warns else "",
              msgs = if (length(msgs)) msgs else "")
  for (nm in c("SpaNorm", "SpaNormUnpolished", "SpaNormNull", "SpaNormNullUnpolished")) {
    if (!is.null(md[[nm]])) out[[nm]] <- pack_fit(md[[nm]])
  }
  out$svg_cols <- intersect(c("svg.F", "svg.p", "svg.fdr"), colnames(rowData(spe)))
  if (!length(out$svg_cols)) out$svg_cols <- ""
  out
}

run <- function(spe, ...) {
  warns <- character(); msgs <- character()
  res <- withCallingHandlers(
    polishSpaNorm(spe, ...),
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") },
    message = function(m) { msgs <<- c(msgs, trimws(conditionMessage(m))); invokeRestart("muffleMessage") }
  )
  pack_run(res, warns, msgs)
}

out <- list(Y = Y, coords = xy, genes = rownames(Y))

## ---- fixture a: no batch, SpaNormSVG first so the null fit exists --------------
set.seed(20260925)
spe <- suppressWarnings(SpaNorm(mk_spe(Y, xy), df.tps = 2L, sample.p = 0.25,
                                verbose = FALSE, backend = "cpu"))
spe <- suppressWarnings(SpaNormSVG(spe, backend = "cpu", verbose = FALSE))
md <- S4Vectors::metadata(spe)
a <- list(input = pack_fit(md$SpaNorm), input_null = pack_fit(md$SpaNormNull),
          svg_F = rowData(spe)$svg.F)
a$default <- run(spe, verbose = FALSE)
a$tight <- run(spe, tol = 1e-12, verbose = FALSE)
a$joint <- run(spe, ls = "joint", verbose = FALSE)
a$profile <- run(spe, psi.method = "profile", verbose = FALSE)
a$profile_joint <- run(spe, psi.method = "profile", ls = "joint", verbose = FALSE)
a$cells_fit <- run(spe, cells = "fit", verbose = FALSE)
a$no_null <- run(spe, null = FALSE, verbose = FALSE)
a$pearson <- run(spe, adj.method = "pearson", scale.factor = 2, verbose = FALSE)
a$maxit3 <- run(spe, maxit = 3L, tol = 1e-12, verbose = FALSE)
# polish twice: the overwrite path starts from the stored unpolished fit
once <- suppressWarnings(polishSpaNorm(spe, psi.method = "profile", verbose = FALSE))
a$overwrite <- run(once, overwrite = TRUE, verbose = FALSE)
# a polished full fit alone, then an SVG-less object: refused without overwrite
a$refused_msg <- tryCatch(polishSpaNorm(once, verbose = FALSE), error = conditionMessage)
# an all-zero gene
cnt0 <- Y; cnt0[1, ] <- 0L
spe0 <- spe; assay(spe0, "counts") <- cnt0
a$zero_gene <- run(spe0, null = FALSE, verbose = FALSE)
a$zero_gene_joint <- run(spe0, ls = "joint", null = FALSE, verbose = FALSE)
out$a <- a

## ---- fixture b: two-level batch, gene 5 absent from level s2 ------------------
set.seed(20260926)
bg <- 5L
b <- factor(rep(c("s1", "s2"), each = NC / 2))
cntb <- Y; cntb[bg, b == "s2"] <- 0L
speb <- mk_spe(cntb, xy)
speb <- suppressWarnings(SpaNorm(speb, df.tps = 2L, sample.p = 0.5, batch = b,
                                 verbose = FALSE, backend = "cpu"))
mdb <- S4Vectors::metadata(speb)
bl <- list(Y = cntb, batch = as.character(b), gene = bg, input = pack_fit(mdb$SpaNorm))
bl$default <- run(speb, null = FALSE, verbose = FALSE)
bl$joint <- run(speb, ls = "joint", null = FALSE, verbose = FALSE)
bl$tight <- run(speb, tol = 1e-12, null = FALSE, verbose = FALSE)
out$b <- bl

## ---- collinear unpenalised block: one-hot batch matrix, no intercept -----------
set.seed(20260925)
bm <- factor(rep(c("s1", "s2"), length.out = 400))
spec <- mk_spe(Y[seq_len(20), seq_len(400)], xy[seq_len(400), ])
spec <- suppressWarnings(SpaNorm(spec, df.tps = 2L, sample.p = 0.5,
                                 batch = stats::model.matrix(~ 0 + bm),
                                 verbose = FALSE, backend = "cpu"))
mdc <- S4Vectors::metadata(spec)
cl <- list(Y = Y[seq_len(20), seq_len(400)], coords = xy[seq_len(400), ],
           batch = unname(stats::model.matrix(~ 0 + bm)), input = pack_fit(mdc$SpaNorm))
cl$default <- run(spec, null = FALSE, verbose = FALSE)
cl$joint <- run(spec, ls = "joint", null = FALSE, verbose = FALSE)
out$c <- cl

write_ref(out, "polish_spanorm")
