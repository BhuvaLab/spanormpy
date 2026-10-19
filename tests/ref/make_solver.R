# Reference values for the penalised NB Newton solvers (dense, blocked, compact,
# batch), the batched gram / Schur absorption and nbBlockDesign.
source("tests/ref/_common.R")
ns <- function(n) get(n, envir = asNamespace("SpaNorm"))
newton <- ns(".newtonSolver"); newton_batch <- ns(".newtonSolverBatch")
compact <- ns(".newtonSolverCompact")

set.seed(2)
n <- 200; px <- 5; G <- 3; b <- 5
X <- cbind(1, matrix(rnorm(n * (px - 1)), n, px - 1))
grp <- c(seq_len(G), sample(seq_len(G), n - G, replace = TRUE))
Z <- matrix(0, n, G); Z[cbind(seq_len(n), grp)] <- 1
W <- cbind(X, Z)                                   # 200 x 8
nested <- c(rep(FALSE, px), rep(TRUE, G))
pen <- c(0, 0.1, 0, 0.05, 0, rep(0.3, G))
wt <- matrix(runif(b * n, 0.2, 2), b, n)
wt0 <- wt; wt0[2, ] <- 0                           # gene 2 carries no weight
Sc <- matrix(rnorm(b * ncol(W)), b, ncol(W))
out <- list(n = n, px = px, G = G, W = W, nested = as.integer(nested), pen = pen,
            wt = wt, wt0 = wt0, Sc = Sc)

# fill a (b x k) matrix / (b x k x k) array from per-gene results; NULL -> NA + flag
per_gene <- function(fn, k, wts = wt, dims = k) {
  res <- array(NA_real_, c(nrow(wts), dims)); isnull <- integer(nrow(wts))
  for (g in seq_len(nrow(wts))) {
    r <- fn(g, wts[g, ])
    if (is.null(r)) isnull[g] <- 1L
    else if (length(dims) == 1L) res[g, ] <- r
    else res[g, , ] <- r
  }
  list(val = res, null = isnull)
}

# --- dense (no absorption) -------------------------------------------------
sd <- newton(W, pen, NULL)
r <- per_gene(function(g, w) sd$solve(w, Sc[g, ]), ncol(W), wt0)
out$dense_step <- r$val; out$dense_step_null <- r$null
r <- per_gene(function(g, w) sd$xcov(w), 0, wt0, c(ncol(W), ncol(W)))
out$dense_xcov <- r$val; out$dense_xcov_null <- r$null
r <- per_gene(function(g, w) sd$factor(w)$info, 0, wt, c(ncol(W), ncol(W)))
out$dense_info <- r$val
# a state reused by solve/xcov
st <- sd$factor(wt[1, ])
out$dense_step_state <- sd$solve(st, Sc[1, ]); out$dense_xcov_state <- sd$xcov(st)

# --- nested indicator absorption (diagonal C) -----------------------------
sn <- newton(W, pen, nested)
r <- per_gene(function(g, w) sn$solve(w, Sc[g, ]), ncol(W), wt)
out$nested_step <- r$val
r <- per_gene(function(g, w) sn$xcov(w), 0, wt, c(px, px)); out$nested_xcov <- r$val
r <- per_gene(function(g, w) sn$factor(w)$S, 0, wt, c(px, px)); out$nested_S <- r$val
r <- per_gene(function(g, w) unname(sn$factor(w)$B), 0, wt, c(px, G)); out$nested_B <- r$val
r <- per_gene(function(g, w) unname(sn$factor(w)$cvec), G, wt); out$nested_cvec <- r$val
# integer grouping with the same meaning as the logical
sg <- newton(W, pen, c(rep(NA_integer_, px), seq_len(G)))
out$nested_step_blk <- sg$solve(wt[3, ], Sc[3, ])   # 1x1 blocks via the blocked path

# --- batch solver and batched helpers ------------------------------------
sb <- newton_batch(W, pen, nested)
stb <- sb$factor(wt); out$batch_step <- sb$solve(stb, Sc)
out$batch_xcov <- sb$xcov(stb); out$batch_S <- stb$S; out$batch_B <- stb$B
out$batch_cvec <- stb$cvec
stb0 <- sb$factor(wt0); out$batch0_ok <- as.integer(stb0$ok)
out$batch0_step <- sb$solve(stb0, Sc); out$batch0_xcov <- sb$xcov(stb0)
sbd <- newton_batch(W, pen, rep(FALSE, ncol(W)))
std <- sbd$factor(wt); out$batchd_step <- sbd$solve(std, Sc); out$batchd_xcov <- sbd$xcov(std)
std0 <- sbd$factor(wt0); out$batchd0_ok <- as.integer(std0$ok)
out$batchd0_step <- sbd$solve(std0, Sc)
out$gram_pen <- nbGramBatch(W, wt, pen)
out$gram_nopen <- nbGramBatch(W, wt)
out$gram_tile <- nbGramBatch(W, wt, pen, cell.tile = 7)
out$absorb_S <- nbAbsorbGramBatch(W, pen, nested, wt)
pp <- nbAbsorbGramBatch(W, pen, nested, wt, parts = TRUE)
out$absorb_parts_S <- pp$S; out$absorb_parts_B <- pp$B; out$absorb_parts_cvec <- pp$cvec
out$absorb_parts_xi <- pp$xi; out$absorb_parts_zi <- pp$zi
out$absorb_tile <- nbAbsorbGramBatch(W, pen, nested, wt, cell.tile = 7)
# a group holding no cells: absorption must keep its (zero-weight) slot
grp_e <- sample(setdiff(seq_len(G), 2L), n, replace = TRUE)
Ze <- matrix(0, n, G); Ze[cbind(seq_len(n), grp_e)] <- 1
We <- cbind(X, Ze); out$We <- We
pp <- nbAbsorbGramBatch(We, pen, nested, wt, parts = TRUE)
out$absorb_empty_S <- pp$S; out$absorb_empty_cvec <- pp$cvec; out$absorb_empty_B <- pp$B

# --- compact block design and the blocked (per-sample) absorption ---------
make_block <- function(sizes, seed) {
  set.seed(seed)
  lev <- sprintf("P%02d", seq_along(sizes))
  blk <- factor(rep(lev, sizes), levels = lev)
  nn <- length(blk); blk <- blk[sample.int(nn)]
  Xb <- matrix(rnorm(nn * 3), nn, 3, dimnames = list(NULL, paste0("n", 1:3)))
  l <- rnorm(nn, sd = 0.5)
  Zb <- cbind(int = 1, ls1 = l * rnorm(nn))
  list(X = Xb, Z = Zb, block = blk, D = nbBlockDesign(Xb, Zb, blk))
}
blockcase <- function(tag, f, pen_xz, wts) {
  D <- f$D; nn <- nrow(D); pcol <- ncol(D)
  pen_full <- ns(".blockExpand")(D, pen_xz, "pen")
  sc <- compact(D, pen_full)
  Scb <- matrix(rnorm(nrow(wts) * pcol), nrow(wts), pcol)
  o <- list()
  o$X <- f$X; o$Z <- f$Z; o$block <- as.character(f$block)
  o$levels <- levels(f$block); o$pen_xz <- pen_xz; o$wts <- wts; o$Sc <- Scb
  o$dense <- as.matrix(D); o$colnames <- colnames(D)
  o$step <- NULL; o$S <- NULL
  res <- list(step = array(NA_real_, c(nrow(wts), pcol)),
              xcov = array(NA_real_, c(nrow(wts), D$px, D$px)),
              S = array(NA_real_, c(nrow(wts), D$px, D$px)),
              B = array(NA_real_, c(nrow(wts), D$px, pcol - D$px)),
              rank = matrix(NA_integer_, nrow(wts), D$G),
              Cinv = array(NA_real_, c(nrow(wts), D$G, D$q, D$q)))
  for (g in seq_len(nrow(wts))) {
    w <- wts[g, ]; st <- sc$factor(w)
    res$step[g, ] <- sc$solve(st, Scb[g, ]); res$xcov[g, , ] <- sc$xcov(st)
    res$S[g, , ] <- st$S; res$B[g, , ] <- st$B; res$rank[g, ] <- st$rank
    for (k in seq_len(D$G)) res$Cinv[g, k, , ] <- tcrossprod(st$H[[k]])
  }
  o$step <- res$step; o$xcov <- res$xcov; o$S <- res$S; o$B <- res$B
  o$rank <- res$rank; o$Cinv <- res$Cinv
  # design helpers
  set.seed(9)
  a <- rnorm(pcol); r <- rnorm(nn)
  A3 <- matrix(rnorm(3 * pcol), 3); R3 <- matrix(rnorm(3 * nn), 3)
  A8 <- matrix(rnorm(8 * pcol), 8); R8 <- matrix(rnorm(8 * nn), 8)
  o$a <- a; o$r <- r; o$A3 <- A3; o$R3 <- R3; o$A8 <- A8; o$R8 <- R8
  o$eta_vec <- ns(".blockEtaVec")(D, a); o$score_vec <- ns(".blockScoreVec")(D, r)
  o$eta3 <- ns(".blockEtaBatch")(D, A3); o$score3 <- ns(".blockScoreBatch")(R3, D)
  o$eta8 <- ns(".blockEtaBatch")(D, A8); o$score8 <- ns(".blockScoreBatch")(R8, D)
  out[[tag]] <<- o
}
f1 <- make_block(c(150, 60, 220, 35, 90), 1)
set.seed(21); w1 <- matrix(runif(3 * nrow(f1$X), 0.05, 5), 3)
blockcase("blk_main", f1, c(0, 0, 0, 0, 1e-3), w1)
# patient 2 has two cells and unpenalised block columns: rank 2 (eigen route)
f2 <- make_block(c(100, 2, 98), 6)
set.seed(22); w2 <- matrix(runif(2 * nrow(f2$X), 0.1, 2), 2)
blockcase("blk_aliased", f2, rep(0, 5), w2)
# a level with no cells, all block columns penalised
f3 <- make_block(c(80, 0, 70, 50), 5)
set.seed(23); w3 <- matrix(runif(2 * nrow(f3$X), 0.1, 2), 2)
blockcase("blk_empty", f3, c(rep(0, 3), rep(1e-3, 2)), w3)

# dense per-sample (blocked) absorption on the main design
Wb <- as.matrix(f1$D); grpb <- ns(".blockDenseAbsorb")(f1$D)
penb <- ns(".blockExpand")(f1$D, c(0, 0, 0, 0.5, 0.5), "pen")
sbk <- newton(Wb, penb, grpb)
set.seed(31); sb_w <- runif(nrow(Wb), 0.05, 5); sb_s <- rnorm(ncol(Wb))
stk <- sbk$factor(sb_w)
out$blocked <- list(W = Wb, group = ifelse(is.na(grpb), 0L, grpb), pen = penb, w = sb_w,
                    s = sb_s, step = sbk$solve(sb_w, sb_s), xcov = sbk$xcov(sb_w),
                    S = stk$S, B = unname(stk$B))
write_ref(out, "solver")
