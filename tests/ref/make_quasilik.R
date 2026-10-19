# Reference values for quasi-likelihood dispersion (quasilik.R).
source("tests/ref/_common.R")
suppressPackageStartupMessages(library(SingleCellExperiment))
ns <- function(n) get(n, envir = asNamespace("SpaNorm"))
nbUnitDeviance <- ns("nbUnitDeviance"); nbDevianceMoments <- ns("nbDevianceMoments")
qlMomentTable <- ns("qlMomentTable"); qlDispersion <- ns("qlDispersion"); qlInterp <- ns(".qlInterp")
fitSpaNormNB <- ns("fitSpaNormNB"); bs.tps <- ns("bs.tps")
quiet <- function(...) NULL
out <- list()

# --- nbUnitDeviance (testthat case 1) ---------------------------------------
set.seed(1)
ng <- 6L; n <- 50L
mu <- matrix(exp(rnorm(ng * n, log(0.5), 1.5)), ng, n)
phi <- c(0.05, 0.2, 1, 5, 20, 5)
y <- matrix(stats::rnbinom(ng * n, mu = mu, size = 1 / phi), ng, n)
y[1, 1:10] <- 0
out$ud <- list(y = y, mu = mu, phi = phi, dev = nbUnitDeviance(y, mu, phi),
               dev_scalar = nbUnitDeviance(y, mu, 0.3),
               edger = t(sapply(seq_len(ng), function(g) edgeR::nbinomUnitDeviance(y[g, ], mu[g, ], phi[g]))))
# zero mu, zero y
out$ud$edge <- nbUnitDeviance(matrix(c(0, 3, 0, 2), 2, 2), matrix(c(0, 0, 1, 1.5), 2, 2), c(0.5, 2))

# --- nbDevianceMoments -------------------------------------------------------
g <- expand.grid(mu = c(0.1, 1, 10, 100, 1000), phi = c(0.01, 0.2, 1, 5, 20))
a <- nbDevianceMoments(g$mu, g$phi, eps = 1e-10)
b <- nbDevianceMoments(g$mu, g$phi, eps = 1e-13)
out$mom <- list(mu = g$mu, phi = g$phi, w0 = a$w0, w1 = a$w1, w0_tight = b$w0, w1_tight = b$w1)
m1 <- nbDevianceMoments(mu = 500, phi = 1e-6)
m2 <- nbDevianceMoments(mu = 0.1, phi = 5)
out$mom$near_poisson <- c(m1$w0, m1$w1); out$mom$sparse <- c(m2$w0, m2$w1)
# mixed vector incl. invalid entries (mu = 0, phi = 0, phi = Inf, NaN) and a maxterms cap
bad_mu <- c(0, 1e-40, 2, 3, NaN, 5, 40)
bad_phi <- c(1, 1, 0, Inf, 1, 0.5, 3)
bm <- nbDevianceMoments(bad_mu, bad_phi)
out$mom_bad <- list(mu = bad_mu, phi = bad_phi, w0 = bm$w0, w1 = bm$w1)
cm <- nbDevianceMoments(c(50, 300), c(0.5, 20), maxterms = 25L)
out$mom_cap <- list(w0 = cm$w0, w1 = cm$w1)
# a vector long enough to need several width groups, scalar phi recycled
mu_v <- exp(seq(log(1e-3), log(5e3), length.out = 41))
mv <- nbDevianceMoments(mu_v, 2.5)
out$mom_vec <- list(mu = mu_v, w0 = mv$w0, w1 = mv$w1)

# --- qlMomentTable and .qlInterp ---------------------------------------------
tabs <- list(
  default = qlMomentTable(log(c(1e-6, 1e3)), log(c(0.1, 10))),
  one_phi = qlMomentTable(log(c(0.01, 50)), log(2), step_mu = 0.3),
  steps = qlMomentTable(log(c(0.05, 20)), log(c(0.05, 8)), step_mu = 0.5, step_phi = 0.7),
  grid = qlMomentTable(log(c(0.05, 20)), log(c(0.05, 8)), ngrid = 17, nphi = 5),
  degenerate = qlMomentTable(log(c(2, 2)), log(c(1, 1.5)), ngrid = 4, nphi = 3)
)
set.seed(21)
interp <- list()
for (nm in names(tabs)) {
  tb <- tabs[[nm]]
  out[[paste0("tab_", nm)]] <- tb
  lmu <- matrix(runif(7 * 30, tb$lo - 1.5, tb$lo + tb$h * (tb$K - 1) + 1.5), 7, 30)
  lphi <- if (tb$J > 1) runif(7, tb$plo - 0.5, tb$plo + tb$ph * (tb$J - 1) + 0.5) else rep(tb$plo, 7)
  w <- qlInterp(tb, lmu, lphi)
  out[[paste0("interp_", nm)]] <- list(lmu = lmu, lphi = lphi, w0 = w$w0, w1 = w$w1)
}

# --- qlDispersion on a dlpfc_small SpaNorm fit -------------------------------
spe <- zellkonverter::readH5AD("tests/ref/data/dlpfc_small.h5ad")
ngd <- 40
Y <- as.matrix(assay(spe, "counts"))[seq_len(ngd), ]
nc <- ncol(Y)
xy <- reducedDim(spe, "spatial")
coords <- apply(xy, 2, \(x) (x - min(x)) / (max(x) - min(x)) - 0.5)
bs.bio <- bs.tps(coords[, 1], coords[, 2], df.tps = 2)
bs.ls <- bs.tps(coords[, 1], coords[, 2], df.tps = 1)
logLS <- log(pmax(1e-08, colSums(Y)))
W <- model.matrix(~ logLS + bs.bio + logLS:bs.ls)[, -1]
dimnames(W) <- NULL
df <- c(as.integer(attr(bs.bio, "df.tps")), as.integer(attr(bs.ls, "df.tps")))
lam <- rep(0, ncol(W)); lam[seq(2, prod(df[1:2]) + 1)] <- 1e-4; lam <- lam[-1] * nc
set.seed(1); idx <- rep(FALSE, nc); idx[sample.int(nc, 1000)] <- TRUE
set.seed(11)
fa <- fitSpaNormNB(Y, W, idx, maxit.psi = 3, maxit.nb = 10, tol = 1e-4, is.spanorm = TRUE, lambda.a = lam,
                   backend = "cpu", msgfun = quiet)
MU <- SpaNorm::calculateMu(fa$gmean, fa$alpha, W)
PHI <- fa$psi
DES <- cbind(1, W)
out$ql <- list(Y = Y, mu = MU, phi = PHI, design = DES)
for (lev in c("trace", "exact")) for (mo in c("table", "grid", "cell")) {
  r <- qlDispersion(Y, MU, PHI, design = DES, leverage = lev, moments = mo)
  out$ql[[paste(lev, mo, sep = "_")]] <- r
}
out$ql$prior <- qlDispersion(Y, MU, PHI, design = DES, prior = 1.7, leverage = "exact", moments = "cell")
out$ql$prior_table <- qlDispersion(Y, MU, PHI, design = DES, prior = 1.7)
out$ql$prior_grid <- qlDispersion(Y, MU, PHI, p = ncol(DES), prior = 1.7, moments = "grid", ngrid = 64L)
out$ql$small_grid <- qlDispersion(Y, MU, PHI, p = ncol(DES), moments = "table", ngrid = 32L, nphi = 8L)
tab <- qlMomentTable(lmu_range = log(c(1e-8, max(Y) + 1)), lphi_range = log(range(PHI)))
out$ql$with_table <- qlDispersion(Y, MU, PHI, p = ncol(DES), table = tab)
out$ql$scalar_phi <- qlDispersion(Y, MU, 2, p = 3L)
out$ql$tiny_n <- qlDispersion(Y[, 1:3], MU[, 1:3], PHI, p = 3L)   # hdp < 1e-4 early return
out$ql$full_table <- tab

# --- edgeR oracle (testthat case) ---------------------------------------------
set.seed(11)
eg <- 40L; en <- 8L
grp <- factor(rep(c("a", "b"), each = en / 2))
edes <- stats::model.matrix(~ grp)
mu0 <- rep(c(2, 20, 200, 2000), length.out = eg)
ey <- matrix(stats::rnbinom(eg * en, mu = mu0, size = 1 / 0.2), eg, en)
fit <- edgeR::glmQLFit(ey, design = edes, dispersion = 0.2, legacy = FALSE, abundance.trend = FALSE)
got <- qlDispersion(ey, mu = fit$fitted.values, phi = 0.2, design = edes,
                    prior = fit$average.ql.dispersion, leverage = "exact", moments = "cell")
out$edger <- list(y = ey, mu = fit$fitted.values, design = edes, prior = fit$average.ql.dispersion,
                  df = as.numeric(fit$df.residual.adj), deviance = as.numeric(fit$deviance.adj),
                  got = got)

# --- testthat random cases (same seeds as the R tests) ------------------------
set.seed(12); ng <- 20L; n <- 400L
d12 <- cbind(1, rnorm(n), rnorm(n))
y12 <- matrix(stats::rnbinom(ng * n, mu = 30, size = 1 / 0.3), ng, n)
mu12 <- matrix(30, ng, n)
out$t12 <- list(y = y12, mu = mu12, design = d12,
                ex = qlDispersion(y12, mu12, phi = 0.3, design = d12, leverage = "exact", moments = "cell"),
                ap = qlDispersion(y12, mu12, phi = 0.3, design = d12, leverage = "trace", moments = "cell"))
set.seed(14); ng <- 24L; n <- 500L
d14 <- cbind(1, rnorm(n))
mu14 <- matrix(exp(rnorm(ng * n, log(0.1), 1.2)), ng, n)
phi14 <- exp(seq(log(0.05), log(30), length.out = ng))
y14 <- matrix(stats::rnbinom(ng * n, mu = mu14, size = 1 / phi14), ng, n)
out$t14 <- list(y = y14, mu = mu14, phi = phi14, design = d14,
                ex = qlDispersion(y14, mu14, phi = phi14, design = d14, moments = "cell"),
                tb = qlDispersion(y14, mu14, phi = phi14, design = d14, moments = "table"),
                gr = qlDispersion(y14, mu14, phi = 5, design = d14, moments = "grid"),
                tb1 = qlDispersion(y14, mu14, phi = 5, design = d14, moments = "table"))
set.seed(17); ng <- 20L; n <- 300L
mu17 <- matrix(exp(rnorm(ng * n, log(0.3), 1.4)), ng, n)
phi17 <- exp(seq(log(0.05), log(20), length.out = ng))
y17 <- matrix(stats::rnbinom(ng * n, mu = mu17, size = 1 / phi17), ng, n)
tab17 <- qlMomentTable(lmu_range = log(c(1e-8, max(y17) + 1)), lphi_range = log(range(phi17)))
out$t17 <- list(y = y17, mu = mu17, phi = phi17, tab = tab17,
                whole = qlDispersion(y17, mu17, phi17, p = 3L, table = tab17),
                ex = qlDispersion(y17, mu17, phi17, p = 3L, moments = "cell"))
write_ref(out, "quasilik")
