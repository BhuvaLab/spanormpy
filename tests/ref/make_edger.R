# Reference values for the edgeR::estimateDisp path SpaNorm uses:
#   estimateDisp(y, design = matrix(1, n), offset = offs, tagwise = TRUE, robust = TRUE)
# Cases: (a) simulated counts, zero offset; (b) genes x cells offset and counts captured from a
# real SpaNorm (fitNB) outer iteration on dlpfc data; (c) all-zero / near-constant genes;
# (d) robust = FALSE; (e) cells with a very low fitted mean, giving unequal residual df.
source("tests/ref/_common.R")
suppressPackageStartupMessages(library(edgeR))

# the adjusted profile likelihood grid exactly as estimateDisp.default builds it
apl_grid <- function(y, offs, min.row.sum = 5, grid.length = 21, grid.range = c(-10, 10)) {
  sel <- rowSums(y) >= min.row.sum
  sely <- y[sel, , drop = FALSE]; seloffset <- offs[sel, , drop = FALSE]
  pts <- seq(grid.range[1], grid.range[2], length.out = grid.length)
  disp <- 0.1 * 2^pts
  l0 <- matrix(0, sum(sel), grid.length)
  design <- matrix(1, ncol(y), 1)
  glmfit <- glmFit(sely, design, offset = seloffset, dispersion = 0.05, prior.count = 0)
  zerofit <- (glmfit$counts < 1e-04 & glmfit$fitted.values < 1e-04)
  for (subg in edgeR:::.comboGroups(zerofit)) {
    cur.nzero <- !zerofit[subg[1], ]
    if (!any(cur.nzero)) next
    if (all(cur.nzero)) redesign <- design else {
      redesign <- design[cur.nzero, , drop = FALSE]
      QR <- qr(redesign); redesign <- redesign[, QR$pivot[1:QR$rank], drop = FALSE]
      if (nrow(redesign) == ncol(redesign)) next
    }
    cury <- y[sel, , drop = FALSE][subg, cur.nzero, drop = FALSE]
    curo <- seloffset[subg, cur.nzero, drop = FALSE]
    last.beta <- NULL
    for (i in seq_len(grid.length)) {
      out <- adjustedProfileLik(disp[i], y = cury, design = redesign, offset = curo,
                                weights = NULL, start = last.beta, get.coef = TRUE)
      l0[subg, i] <- out$apl; last.beta <- out$beta
    }
  }
  l0
}

run_case <- function(y, offs, robust = TRUE) {
  d <- estimateDisp(y, matrix(1, ncol(y)), offset = offs, tagwise = TRUE, robust = robust)
  list(y = y, offset = offs, robust = as.integer(robust),
       tagwise = d$tagwise.dispersion, common = d$common.dispersion,
       trended = d$trended.dispersion, prior_df = d$prior.df, prior_n = d$prior.n,
       span = d$span, apl = apl_grid(y, offs), sel = as.integer(rowSums(y) >= 5),
       ave_log_cpm = aveLogCPM(y, lib.size = colSums(y), dispersion = d$common.dispersion))
}

out <- list()
# (a)
sim <- sim_counts(100, 300, 3)
y_a <- sim$counts; storage.mode(y_a) <- "double"
out$a <- run_case(y_a, matrix(0, nrow(y_a), ncol(y_a)))
# (d)
out$d <- run_case(y_a, matrix(0, nrow(y_a), ncol(y_a)), robust = FALSE)

# (b) capture the actual estimateDisp arguments of a real fitNB outer loop on dlpfc data
suppressPackageStartupMessages(library(SpatialExperiment))
load("../SpaNorm/data/HumanDLPFC.rda")
spe <- get("HumanDLPFC")
spe <- spe[head(which(filterGenes(spe, prop = 0.2)), 300), ]
set.seed(11)
cells <- sort(sample(ncol(spe), 800))
Y <- as.matrix(counts(spe))[, cells]; storage.mode(Y) <- "double"
xy <- scale(spatialCoords(spe)[cells, ])
W <- cbind(xy[, 1], xy[, 2], xy[, 1]^2, xy[, 1] * xy[, 2], xy[, 2]^2)
lib <- colSums(Y)
offs_fix <- outer(rnorm(nrow(Y), 0, 0.2), rep(1, ncol(Y))) + rep(1, nrow(Y)) %o% (0.5 * log(lib / mean(lib)))
captured <- list()
trace(edgeR:::estimateDisp.default,
      tracer = quote(assign("captured_calls",
                            c(get0("captured_calls", globalenv(), ifnotfound = list()),
                              list(list(y = y, offset = offset))), globalenv())),
      print = FALSE, where = asNamespace("edgeR"))
registerS3method("estimateDisp", "default", get("estimateDisp.default", asNamespace("edgeR")),
                 envir = asNamespace("edgeR"))
set.seed(5)
invisible(SpaNorm::fitNB(Y, W, offset = offs_fix, maxit.psi = 3, maxit.nb = 5, verbose = FALSE, backend = "cpu"))
untrace(edgeR:::estimateDisp.default, where = asNamespace("edgeR"))
cc <- get("captured_calls", globalenv())
stopifnot(length(cc) >= 2)
cat("captured", length(cc), "estimateDisp calls\n")
last <- cc[[length(cc)]]
out$b <- run_case(last$y, last$offset)
out$b$iter_count <- length(cc)

# (c) all-zero, near-constant and tiny genes alongside regular ones
set.seed(9)
y_c <- y_a[1:40, ]
y_c[1, ] <- 0                       # all zero
y_c[2, ] <- 0; y_c[2, 5] <- 1       # row sum 1 (< min.row.sum)
y_c[3, ] <- 0; y_c[3, c(3, 90, 200, 250, 299)] <- 1   # row sum exactly 5
y_c[4, ] <- 3                       # constant
y_c[5, ] <- 0; y_c[5, 1:2] <- c(40, 2)  # almost all zero, two non-zero cells
y_c[6, ] <- rpois(ncol(y_c), 0.05)  # very sparse
out$c <- run_case(y_c, matrix(0, nrow(y_c), ncol(y_c)))

# (e) cells with a very low offset: fitted mean < 1e-4 and zero count -> unequal residual df
set.seed(13)
y_e <- y_a[1:60, ]
offs_e <- matrix(0, 60, ncol(y_e))
low <- sample(300, 40)
offs_e[1:20, low] <- -14
y_e[1:20, low] <- 0
out$e <- run_case(y_e, offs_e)

# (f) Poisson counts: no overdispersion, prior df of the robust fit is Inf
set.seed(21)
mu_f <- exp(rnorm(80, 1.5, 0.8))
y_f <- matrix(rpois(80 * 200, rep(mu_f, 200)), 80, 200); storage.mode(y_f) <- "double"
out$f <- run_case(y_f, matrix(0, 80, 200))
# (g) a few strongly overdispersed outlier genes among well-behaved ones
set.seed(22)
mu_g <- exp(rnorm(120, 2, 0.7))
y_g <- matrix(rnbinom(120 * 150, mu = rep(mu_g, 150), size = 20), 120, 150); storage.mode(y_g) <- "double"
y_g[1:3, ] <- rnbinom(450, mu = mu_g[1:3], size = 0.05)
out$g <- run_case(y_g, matrix(0, 120, 150))
# (h) small problems: few genes / cells, and a per-cell (length ncells) offset
set.seed(23)
y_h <- matrix(rnbinom(12 * 15, mu = 6, size = 3), 12, 15); storage.mode(y_h) <- "double"
offs_h <- matrix(rep(log(colSums(y_h)), each = 12), 12, 15)
out$h <- run_case(y_h, offs_h)
# (i) few selected genes (< 3): squeezeVar returns prior.df = 0
y_i <- y_h[1:2, , drop = FALSE]
out$i <- run_case(y_i, matrix(0, 2, 15))

# (j, k) sparse, strongly overdispersed tiny problems for which the one-group Newton fit fails
# to converge in some APL grid columns; edgeR then returns an unassigned (uninitialised) C local.
# Generator and seeds (5012, 5144) come from a random R-vs-Python fuzz of estimateDisp.
fuzz_case <- function(seed) {
  set.seed(seed)
  G <- sample(c(3, 4, 5, 8, 15, 40, 100, 250), 1)
  n <- sample(c(4, 6, 10, 30, 100, 400, 1000), 1)
  mu <- exp(rnorm(G, sample(c(-1, 0.5, 2), 1), sample(c(0.5, 1.5), 1)))
  lib <- exp(rnorm(n, 0, 0.5))
  size <- sample(c(0.3, 2, 20, 1e4), 1)
  off <- matrix(rnorm(G, 0, 0.3), G, n) + matrix(log(lib), G, n, byrow = TRUE)
  typ <- sample(1:3, 1)
  if (typ >= 2) {
    lowc <- sample(n, max(1, n %/% 5)); lowg <- sample(G, max(1, G %/% 3))
    off[lowg, lowc] <- off[lowg, lowc] - sample(c(8, 14, 25), 1)
  }
  y <- matrix(rnbinom(G * n, mu = exp(off) * mu, size = size), G, n)
  if (typ == 3 && G > 3) { y[sample(G, 1), ] <- 0; y[sample(G, 1), ] <- 0 }
  for (j in which(colSums(y) == 0)) y[sample(G, 1), j] <- 1
  storage.mode(y) <- "double"
  list(y = y, off = off)
}
fz <- fuzz_case(5012); out$j <- run_case(fz$y, fz$off)
fz <- fuzz_case(5144); out$k <- run_case(fz$y, fz$off)
# (z) a cell with zero total counts: aveLogCPM's offset log(lib.size) is -Inf and R errors
y_z <- y_a[1:30, ]; y_z[, 7] <- 0
err_z <- tryCatch({estimateDisp(y_z, matrix(1, ncol(y_z)), offset = matrix(0, 30, ncol(y_z)),
                                tagwise = TRUE, robust = TRUE); ""}, error = function(e) conditionMessage(e))
out$z <- list(y = y_z, offset = matrix(0, 30, ncol(y_z)), robust = 1L, error = err_z)

write_ref(out, "edger")
