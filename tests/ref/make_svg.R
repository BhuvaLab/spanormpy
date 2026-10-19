# Reference values for SpaNormSVG / topSVGs / svgTest (R/mainSpaNormSVG.R).
#
# Sections:
#   base      the Task 10 fit on dlpfc_small (R's size factors, df.tps = 2, tol = 1e-3), then
#             SpaNormSVG; exports the input fit, the technical (null) fit, the SVG table,
#             F.raw and topSVGs.
#   default   R's own default pipeline (SpaNorm() then SpaNormSVG()), with R's size factors and
#             subsets exported so Python can inject them (strict parity), plus two more default
#             runs under other seeds: the R-vs-R seed-to-seed noise floor of svg.F.
#   p1..p4    the paired-null path on the 40 gene x 500 spot polish fixture (same fit as
#             make_polish_spanorm.R fixture a): polished full fit with no null, a stored
#             unpolished null, the reviewer's mismatched-settings sequence, and a profile polish
#             with a non-default psi.range.
#   b         the batch design with one gene absent from a batch level (polish fixture b).
source("tests/ref/_common.R")
suppressPackageStartupMessages({
  library(SpatialExperiment)
  library(SingleCellExperiment)
})

sce <- zellkonverter::readH5AD("tests/ref/data/dlpfc_small.h5ad")
Y <- as.matrix(assay(sce, "counts"))
xy <- as.matrix(reducedDim(sce, "spatial"))
mk_spe <- function(Y, xy) {
  spe <- SpatialExperiment(assays = list(counts = Y), spatialCoords = xy)
  colnames(spe) <- paste0("c", seq_len(ncol(Y)))
  rownames(spe) <- rownames(Y)
  spe
}
spe <- mk_spe(Y, xy)
sf <- as.numeric(rhdf5::h5read("tests/ref/data/spanorm.h5", "sf"))

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

# run an expression, collecting warnings and messages
capture <- function(expr) {
  warns <- character(); msgs <- character()
  res <- withCallingHandlers(expr,
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") },
    message = function(m) { msgs <<- c(msgs, trimws(conditionMessage(m))); invokeRestart("muffleMessage") })
  list(res = res, warns = if (length(warns)) warns else "", msgs = if (length(msgs)) msgs else "")
}

# SpaNormSVG on `s`; exports the stored fits and the SVG table
svg_run <- function(s, seed, full = TRUE) {
  set.seed(seed)
  cp <- capture(SpaNormSVG(s, backend = "cpu", verbose = FALSE))
  r <- cp$res
  md <- S4Vectors::metadata(r)
  out <- list(null = pack_fit(md$SpaNormNull), svg_F = rowData(r)$svg.F, svg_p = rowData(r)$svg.p,
              svg_fdr = rowData(r)$svg.fdr, warns = cp$warns, msgs = cp$msgs)
  if (full) out$full <- pack_fit(md$SpaNorm)
  if (!is.null(md$SpaNormNullUnpolished)) out$null_unpolished <- pack_fit(md$SpaNormNullUnpolished)
  list(spe = r, out = out)
}

out <- list(sf = sf, genes = rownames(Y))

## ---- base: Task 10 fit, SpaNormSVG ---------------------------------------------
spe_sf <- spe
sizeFactors(spe_sf) <- sf
set.seed(10)
base <- suppressMessages(suppressWarnings(
  SpaNorm(spe_sf, df.tps = 2, tol = 1e-3, sample.p = 0.25, verbose = FALSE)))
b <- svg_run(base, 30)
b$out$input <- pack_fit(S4Vectors::metadata(base)$SpaNorm)
# F.raw: SpaNormSVG drops the attribute when it binds the table into rowData, so recompute it
# from the stored pair with R's own svgTest
mdb <- S4Vectors::metadata(b$spe)
sv <- SpaNorm:::svgTest(as.matrix(assay(base, "counts")), mdb$SpaNorm, mdb$SpaNormNull)
b$out$F_raw <- attr(sv, "F.raw")
tp <- topSVGs(b$spe, n = 20, fdr = 0.05)
b$out$top_genes <- rownames(tp)
b$out$top_F <- tp$svg.F
b$out$top_p <- tp$svg.p
b$out$top_fdr <- tp$svg.fdr
tp10 <- topSVGs(b$spe)
b$out$top10_genes <- rownames(tp10)
out$base <- b$out

## ---- default pipeline + seed-to-seed noise floor -------------------------------
dflt_run <- function(seed, seed_svg) {
  set.seed(seed)
  s <- suppressMessages(suppressWarnings(SpaNorm(spe, verbose = FALSE)))
  r <- svg_run(s, seed_svg)
  r$out$input <- pack_fit(S4Vectors::metadata(s)$SpaNorm)
  r$out$sf <- exp(S4Vectors::metadata(s)$SpaNorm@W[, 1])
  r$out
}
out$default <- dflt_run(10, 31)
out$seed11 <- dflt_run(11, 32)[c("svg_F")]
out$seed12 <- dflt_run(12, 33)[c("svg_F")]

## ---- polished fixtures (make_polish_spanorm.R fixture a) -----------------------
NG <- 40L; NC <- 500L
top <- order(-rowSums(Y))[seq_len(NG)]
Ys <- Y[top, seq_len(NC)]
xys <- xy[seq_len(NC), , drop = FALSE]
storage.mode(Ys) <- "integer"
colnames(Ys) <- paste0("c", seq_len(NC))
set.seed(20260925)
sp0 <- suppressWarnings(SpaNorm(mk_spe(Ys, xys), df.tps = 2L, sample.p = 0.25,
                                verbose = FALSE, backend = "cpu"))
pol <- list(Y = Ys, coords = xys, genes = rownames(Ys),
            input = pack_fit(S4Vectors::metadata(sp0)$SpaNorm))
quiet_polish <- function(s, ...) suppressWarnings(polishSpaNorm(s, verbose = FALSE, ...))

# p1: polished full fit, no stored null -> SpaNormSVG fits the null and polishes it
p1 <- svg_run(quiet_polish(sp0, null = FALSE), 40, full = FALSE)
pol$p1 <- p1$out

# unpolished SVG first (stored null), then p2: polish only the full fit; SVG re-polishes the null
s_svg <- svg_run(sp0, 41)
pol$unpolished <- s_svg$out
sp2 <- quiet_polish(s_svg$spe, null = FALSE)
p2 <- svg_run(sp2, 42, full = FALSE)
pol$p2 <- p2$out

# p3: the reviewer's sequence -- polish both (fixed), then the full fit alone (profile)
sp3 <- quiet_polish(s_svg$spe)
sp3 <- quiet_polish(sp3, overwrite = TRUE, null = FALSE, psi.method = "profile")
p3 <- svg_run(sp3, 43, full = FALSE)
pol$p3 <- p3$out
# the same pair polished consistently from scratch
sp3r <- quiet_polish(s_svg$spe, psi.method = "profile")
p3r <- svg_run(sp3r, 44, full = FALSE)
pol$p3_ref <- p3r$out

# p4: profile polish with a non-default psi.range
sp4 <- quiet_polish(sp0, psi.method = "profile", psi.range = c(0.02, 0.2), null = FALSE)
p4 <- svg_run(sp4, 45, full = FALSE)
pol$p4 <- p4$out
# p5: maxit / tol are forwarded to the null
sp5 <- quiet_polish(sp0, null = FALSE, maxit = 7L, tol = 1e-5)
p5 <- svg_run(sp5, 46, full = FALSE)
pol$p5 <- p5$out
# the pre-polish messages of the mismatched / unpolished-stored-null paths
out$pol <- pol

## ---- batch design with a gene absent from one level (fixture b) ---------------
set.seed(20260926)
bg <- 5L
bf <- factor(rep(c("s1", "s2"), each = NC / 2))
cntb <- Ys; cntb[bg, bf == "s2"] <- 0L
speb <- mk_spe(cntb, xys)
speb <- suppressWarnings(SpaNorm(speb, df.tps = 2L, sample.p = 0.5, batch = bf,
                                 verbose = FALSE, backend = "cpu"))
bt <- list(Y = cntb, batch = as.character(bf), gene = bg,
           input = pack_fit(S4Vectors::metadata(speb)$SpaNorm))
spb <- quiet_polish(speb, null = FALSE)
bt$run <- svg_run(spb, 47, full = FALSE)$out
out$batch <- bt

write_ref(out, "svg")
