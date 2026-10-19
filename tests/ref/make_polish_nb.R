# Reference values for polishNB (batch and gene engines, fixed and profile psi,
# offsets, absorb, nbBlockDesign, warm) and nbProfilePsi. Fixtures are the simulated
# ones of tests/testthat/test-polishNB.R (the SpaNorm-fit fixtures of helper-polish.R
# belong to polishSpaNorm, a later task).
source("tests/ref/_common.R")

out <- list(mu_floor = nbMuFloor())
ENGINES <- c("batch", "gene")
METHODS <- c("profile", "fixed")

# run polishNB for every engine x psi.method on one case and store inputs + results
run_case <- function(name, Y, W, A0, psi0, args = list(), W_in = W, methods = METHODS,
                     inputs = list()) {
  out[[name]] <<- c(list(Y = Y, A0 = A0, psi0 = psi0), inputs)
  for (eng in ENGINES) for (m in methods) {
    r <- do.call(polishNB, c(list(Y, W_in, A0, psi0, engine = eng, psi.method = m), args))
    out[[name]][[paste0(eng, "_", m)]] <<- list(
      alpha = r$alpha, psi = r$psi, loglik = r$loglik,
      iterations = r$polish$iterations, restarted = r$polish$restarted,
      capped = r$polish$capped, singular = r$polish$singular,
      psi_bound = r$polish$psi_bound, polished = r$polish$polished)
  }
  invisible(NULL)
}

## ---- plain: a fitNB start, a degenerate-start gene ---------------------------------
set.seed(101)
G <- 20; n <- 300
x1 <- rnorm(n); x2 <- rnorm(n)
W <- cbind(1, x1, x2)
Y <- t(vapply(seq_len(G), function(g)
  rnbinom(n, mu = exp(0.5 + g / 10 + 0.3 * x1), size = 3), numeric(n)))
storage.mode(Y) <- "integer"
fit <- fitNB(Y, W, lambda.a = 0.1, verbose = FALSE, backend = "cpu")
A0 <- fit$alpha; psi0 <- fit$psi
A0[G, ] <- c(-40, 0, 0)                      # degenerate start: restarted
run_case("plain", Y, W, A0, psi0, args = list(lambda.a = 0.1),
         inputs = list(W = W, pen = 0.1))

## ---- offsets: a per-cell vector and a genes x cells matrix --------------------------
set.seed(102)
G2 <- 8; n2 <- 300
W2 <- cbind(1, rnorm(n2))
off <- log(runif(n2, 0.3, 3))
O <- t(vapply(seq_len(G2), function(g) off + g / 4, numeric(n2)))
Y2 <- t(vapply(seq_len(G2), function(g)
  rnbinom(n2, mu = exp(0.5 + 0.3 * W2[, 2] + O[g, ]), size = 4), numeric(n2)))
storage.mode(Y2) <- "integer"
pen2 <- c(0, 0.5)
run_case("offvec", Y2, W2, matrix(0, G2, 2), rep(0.3, G2),
         args = list(lambda.a = pen2, offset = off),
         inputs = list(W = W2, pen = pen2, offset = off))
run_case("offmat", Y2, W2, matrix(0, G2, 2), rep(0.3, G2),
         args = list(lambda.a = pen2, offset = O),
         inputs = list(W = W2, pen = pen2, offset = O))

## ---- absorb: cell-type intercepts, nested indicators, slopes -------------------------
set.seed(20)
Gs <- 6; ns <- 240
smp <- rep(1:4, each = 60)
ct <- rep(c("A", "B"), times = 120)
xs <- rnorm(ns)
Zn <- do.call(cbind, lapply(1:4, function(s)
  cbind(as.numeric(smp == s & ct == "A"), as.numeric(smp == s & ct == "B"))))
Zs <- do.call(cbind, lapply(1:4, function(s) xs * (smp == s)))
Ws <- cbind(A = as.numeric(ct == "A"), B = as.numeric(ct == "B"), x = xs, Zn, Zs)
ps <- ncol(Ws)
u <- rnorm(8, 0, 0.3)
Ys <- t(vapply(seq_len(Gs), function(g)
  rnbinom(ns, mu = exp(0.5 + 0.2 * g * (ct == "A") + 0.3 * xs +
                         u[(smp - 1) * 2 + (ct == "B") + 1]), size = 4), numeric(ns)))
storage.mode(Ys) <- "integer"
pens <- c(0, 0, 0.01, rep(2, 8), rep(5, 4))
starts <- c(TRUE, TRUE, rep(FALSE, ps - 2))
nested <- c(rep(FALSE, 3), rep(TRUE, 8), rep(FALSE, 4))
group <- c(rep(NA, 3), rep(1:4, each = 2), 1:4)
sl_in <- list(W = Ws, pen = pens, start = as.integer(starts),
              nested = as.integer(nested), group = ifelse(is.na(group), -1L, group))
run_case("absorb", Ys, Ws, matrix(0, Gs, ps), rep(0.3, Gs),
         args = list(lambda.a = pens, absorb = nested, start.cols = starts), inputs = sl_in)
run_case("absgrp", Ys, Ws, matrix(0, Gs, ps), rep(0.3, Gs),
         args = list(lambda.a = pens, absorb = group, start.cols = starts), inputs = sl_in)

## ---- nbBlockDesign: per-sample (intercept, slope-free) blocks ------------------------
set.seed(30)
Gb <- 6; nb_ <- 240
sb <- rep(1:4, each = 60)
ctb <- rep(c("A", "B"), times = 120)
xb <- rnorm(nb_)
Xb <- cbind(`(Intercept)` = 1, x = xb)
Zb <- cbind(cA = as.numeric(ctb == "A"), cB = as.numeric(ctb == "B"))
blk <- paste0("s", sb)
Yb <- t(vapply(seq_len(Gb), function(g)
  rnbinom(nb_, mu = exp(0.5 + 0.2 * g * (ctb == "A") + 0.3 * xb), size = 4), numeric(nb_)))
storage.mode(Yb) <- "integer"
D <- nbBlockDesign(Xb, Zb, blk)
penb <- c(0, 0.01, 2, 2)
startb <- c(TRUE, FALSE, FALSE, FALSE)
pb <- ncol(Xb) + 4 * ncol(Zb)
run_case("block", Yb, D, matrix(0, Gb, pb), rep(0.3, Gb),
         args = list(lambda.a = penb, start.cols = startb), W_in = D,
         inputs = list(X = Xb, Z = Zb, block = blk, pen = penb, start = as.integer(startb)))

## ---- warm: re-polish a converged fit at the held dispersion --------------------------
conv <- polishNB(Y[-G, ], W, fit$alpha[-G, ], fit$psi[-G], lambda.a = 0.1,
                 psi.method = "fixed", tol = 1e-12)
out$warm <- list(Y = Y[-G, ], W = W, A0 = conv$alpha, psi0 = conv$psi, pen = 0.1)
for (eng in ENGINES) {
  r <- polishNB(Y[-G, ], W, conv$alpha, conv$psi, lambda.a = 0.2, warm = TRUE, engine = eng)
  out$warm[[eng]] <- list(alpha = r$alpha, psi = r$psi, loglik = r$loglik,
                          iterations = r$polish$iterations, capped = r$polish$capped,
                          polished = r$polish$polished)
}
out$warm$pen_warm <- 0.2

## ---- nbProfilePsi ---------------------------------------------------------------------
set.seed(31)
np_ <- 400
Wp <- cbind(1, rnorm(np_))
offp <- log(runif(np_, 0.3, 3))
Ap <- rbind(c(1, 0.3), c(0.5, -0.2), c(log(5), 0), c(0.2, 0.5))
Yp <- rbind(rnbinom(np_, mu = exp(Wp %*% Ap[1, ]), size = 2),
            rnbinom(np_, mu = exp(Wp %*% Ap[2, ]), size = 5),
            rep(5, np_),
            rnbinom(np_, mu = exp(Wp %*% Ap[4, ] + offp), size = 3))
storage.mode(Yp) <- "integer"
psip <- c(0.7, 0.7, 0.123, 0.5)
out$profile <- list(Y = Yp, W = Wp, A = Ap, psi = psip, offset = offp,
                    plain = nbProfilePsi(Yp, Wp, Ap, psip),
                    off = nbProfilePsi(Yp, Wp, Ap, psip, offset = offp),
                    offmat = nbProfilePsi(Yp, Wp, Ap, psip,
                                          offset = rbind(offp, offp + 0.5, offp, offp)))

write_ref(out, "polish_nb")
