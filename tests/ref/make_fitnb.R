# Reference values for the NB fitting core (fitSpaNormNB / fitNB / fitNBGivenPsi / calculateMu).
source("tests/ref/_common.R")
suppressPackageStartupMessages(library(SingleCellExperiment))
ns <- function(n) get(n, envir = asNamespace("SpaNorm"))
fitSpaNormNB <- ns("fitSpaNormNB"); fitNBGivenPsi <- ns("fitNBGivenPsi")
geneBlockIndices <- ns("geneBlockIndices"); bs.tps <- ns("bs.tps")
quiet <- function(...) NULL

spe <- zellkonverter::readH5AD("tests/ref/data/dlpfc_small.h5ad")
ng <- 40
Yfull <- as.matrix(assay(spe, "counts"))
Y <- Yfull[seq_len(ng), ]
nc <- ncol(Y)
xy <- reducedDim(spe, "spatial")
coords <- apply(xy, 2, \(x) (x - min(x)) / (max(x) - min(x)) - 0.5)

# SpaNorm design (fitSpaNorm, df.tps = 2), library size = colSums (bypasses scran)
bs.bio <- bs.tps(coords[, 1], coords[, 2], df.tps = 2)
bs.ls <- bs.tps(coords[, 1], coords[, 2], df.tps = ceiling(2 / 2))
logLS <- log(pmax(1e-08, colSums(Y)))
W <- model.matrix(~ logLS + bs.bio + logLS:bs.ls)[, -1]
dimnames(W) <- NULL
df <- c(as.integer(attr(bs.bio, "df.tps")), as.integer(attr(bs.ls, "df.tps")))
lam <- rep(0, ncol(W)); lam[seq(2, prod(df[1:2]) + 1)] <- 1e-4; lam <- lam[-1] * nc

out <- list(Y = Y, W = W, lambda_a = lam)

# idx and dispersion subset (R-chosen). nsub = 1000 cells, maxn.psi = 500
set.seed(1); idx <- rep(FALSE, nc); idx[sample.int(nc, 1000)] <- TRUE
nsub <- sum(idx)
set.seed(11); psi_pick <- sample.int(nsub, 500)
out$idx <- idx; out$psi_pick <- psi_pick

pack <- function(f, W, prefix) {
  r <- list(alpha = f$alpha, gmean = f$gmean, psi = f$psi, loglik = f$loglik,
            sampling = as.character(f$sampling))
  r
}

# --- A: SpaNorm model (is.spanorm = TRUE) ------------------------------------
set.seed(11)
fa <- fitSpaNormNB(Y, W, idx, maxit.psi = 3, maxit.nb = 10, tol = 1e-4, is.spanorm = TRUE, lambda.a = lam,
                   backend = "cpu", msgfun = quiet)
stopifnot(identical(which(fa$sampling == "dispersion"), sort(which(idx)[psi_pick])))
out$spanorm <- pack(fa)
out$spanorm$mu <- SpaNorm::calculateMu(fa$gmean, fa$alpha, W)
out$spanorm$mu_inf <- SpaNorm::calculateMu(fa$gmean, fa$alpha, W, winsor = Inf)
out$spanorm$mu_w2 <- SpaNorm::calculateMu(fa$gmean, fa$alpha, W, winsor = 2)

# messages
msgs <- character(); set.seed(11)
invisible(fitSpaNormNB(Y, W, idx, maxit.psi = 2, maxit.nb = 10, tol = 1e-4, is.spanorm = TRUE, lambda.a = lam,
                       backend = "cpu", msgfun = function(...) msgs <<- c(msgs, paste0(...))))
out$spanorm_msgs <- msgs

# --- B: blocked fitNBGivenPsi (3 gene blocks) vs unblocked at A's psi -------------
Ysub <- Y[, idx]; Wsub <- W[idx, ]
for (nb in c(1, 3)) {
  fb <- fitNBGivenPsi(Ysub, Wsub, fa$psi, lambda.a = lam, is.spanorm = TRUE, maxit.nb = 10,
                      blocks = geneBlockIndices(ng, nb), backend = "cpu", msgfun = quiet)
  out[[sprintf("givenpsi_b%d", nb)]] <- list(alpha = fb$alpha, gmean = fb$gmean,
                                             loglik = fb$loglik, loglik_iter = fb$loglik.iter)
}
out$givenpsi_psi <- fa$psi

# --- C: generic fitNB with offset -------------------------------------------
Wg <- cbind(1, scale(logLS)[, 1], coords[, 1], coords[, 2])
dimnames(Wg) <- NULL
O <- outer(seq(-0.3, 0.3, length.out = ng), cos(seq_len(nc) / 50)) +
  matrix(rep(0.2 * scale(logLS)[, 1], each = ng), ng, nc)
out$Wg <- Wg; out$O <- O
set.seed(11)
fo <- SpaNorm::fitNB(Y, Wg, idx = idx, maxit.psi = 3, maxit.nb = 10, offset = O, backend = "cpu", verbose = FALSE)
stopifnot(identical(which(fo$sampling == "dispersion"), sort(which(idx)[psi_pick])))
out$offset <- pack(fo)
out$offset$mu <- SpaNorm::calculateMu(fo$gmean, fo$alpha, Wg, offset = O)
out$offset$mu_inf <- SpaNorm::calculateMu(fo$gmean, fo$alpha, Wg, winsor = Inf, offset = O)
set.seed(11)
fn <- SpaNorm::fitNB(Y, Wg, idx = idx, maxit.psi = 3, maxit.nb = 10, lambda.a = 0.5, winsor = 2, backend = "cpu", verbose = FALSE)
out$generic <- pack(fn)

# --- D: fixed psi ------------------------------------------------------------
p0 <- seq(0.05, 0.6, length.out = ng)
out$p0 <- p0
fp <- SpaNorm::fitNB(Y, Wg, idx = idx, psi = p0, offset = O, maxit.nb = 10, backend = "cpu", verbose = FALSE)
out$fixed_psi <- pack(fp)
fp0 <- SpaNorm::fitNB(Y, Wg, psi = p0, maxit.nb = 10, backend = "cpu", verbose = FALSE)
out$fixed_psi_noidx <- pack(fp0)

# --- E: edgeR failure -> psi left at its initial 0 ---------------------------
# (R: tryCatch gives NA psi, psi stays at rep(0, ng) -> infinite size; record what R does)
E_res <- tryCatch({
  set.seed(11)
  testthat::with_mocked_bindings(
    fitSpaNormNB(Y, W, idx, maxit.psi = 2, maxit.nb = 10, is.spanorm = TRUE, lambda.a = lam, backend = "cpu", msgfun = quiet),
    .package = "edgeR", estimateDisp = function(...) stop("boom"))
}, error = function(e) e)
out$edger_fail_is_error <- as.integer(inherits(E_res, "error"))
if (!inherits(E_res, "error")) {
  out$edger_fail <- pack(E_res)
} else {
  out$edger_fail_msg <- conditionMessage(E_res)
}

# --- F: runs to full convergence (default maxit.nb / maxit.psi) --------------
collect <- function() { m <- character(); list(f = function(...) m <<- c(m, paste0(...)), get = function() m) }
# (a) small generic fit, 10 genes x 300 cells
set.seed(5); gng <- 10; gnc <- 300
Wc <- cbind(1, rnorm(gnc), rnorm(gnc)); Ac <- cbind(rnorm(gng, 1.5, .3), rnorm(gng, 0, .5), rnorm(gng, 0, .5))
Yc <- matrix(rnbinom(gng * gnc, mu = exp(Ac %*% t(Wc)), size = 5), gng, gnc)
mc <- collect(); set.seed(7)
fc <- fitSpaNormNB(Yc, Wc, rep(TRUE, gnc), is.spanorm = FALSE, lambda.a = 0, backend = "cpu", msgfun = mc$f)
out$conv_generic <- c(list(Y = Yc, W = Wc, msgs = mc$get()), pack(fc))
# (b) SpaNorm model on a 15 gene x 400 cell dlpfc subset (df.tps = 2)
set.seed(3); cells <- sort(sample(ncol(Yfull), 400))
Y4 <- Yfull[1:15, cells]; n4 <- 400
c4 <- apply(xy[cells, ], 2, \(x) (x - min(x)) / (max(x) - min(x)) - 0.5)
b4 <- bs.tps(c4[, 1], c4[, 2], df.tps = 2); l4 <- bs.tps(c4[, 1], c4[, 2], df.tps = 1)
lls <- log(pmax(1e-08, colSums(Y4)))
W4 <- model.matrix(~ lls + b4 + lls:l4)[, -1]; dimnames(W4) <- NULL
d4 <- c(as.integer(attr(b4, "df.tps")), as.integer(attr(l4, "df.tps")))
lam4 <- rep(0, ncol(W4)); lam4[seq(2, prod(d4[1:2]) + 1)] <- 1e-4; lam4 <- lam4[-1] * n4
m4 <- collect(); set.seed(9)
f4 <- fitSpaNormNB(Y4, W4, rep(TRUE, n4), is.spanorm = TRUE, lambda.a = lam4, backend = "cpu", msgfun = m4$f)
out$conv_spanorm <- c(list(Y = Y4, W = W4, lambda_a = lam4, msgs = m4$get()), pack(f4))
# (c) step halving: fitNBGivenPsi from overshooting starts (psi fixed, winsor = Inf)
out$halv_psi <- rep(0.2, gng)
for (sd in 1:2) {
  set.seed(sd); A0 <- matrix(rnorm(gng * 3, 0, 0.3), gng, 3)
  for (sp in c(FALSE, TRUE)) {
    mh <- collect()
    fh <- fitNBGivenPsi(Yc, Wc, out$halv_psi, lambda.a = if (sp) rep(0.1, 2) else 0.1,
                        gmean = rep(0, gng), alpha = A0, is.spanorm = sp, winsor = Inf,
                        backend = "cpu", msgfun = mh$f)
    out[[sprintf("halv_s%d_sp%d", sd, sp)]] <- list(A0 = A0, alpha = fh$alpha, gmean = fh$gmean,
      loglik = fh$loglik, loglik_iter = fh$loglik.iter, msgs = mh$get())
  }
}

write_ref(out, "fitnb")
