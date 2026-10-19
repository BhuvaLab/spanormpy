source("tests/ref/_common.R")

# ---- geneBlockCount / geneBlockIndices ------------------------------------
# geneBlockCount returns 1L unless an accelerator is in use; pin the cached
# device so the reference reflects a CUDA (8-byte) and an MPS (4-byte) device
# regardless of this machine's hardware.
env <- SpaNorm:::.spanorm_env
grid <- expand.grid(
  n_genes = c(1, 5, 100, 1000, 20000),
  n_cells = c(10, 1000, 1e5),
  n_cov = c(3, 20),
  budget = c(1, 1e6, 1e9, 1e12, Inf)
)
count_for <- function(device, backend) {
  assign("device", device, envir = env)
  on.exit(SpaNorm::resetGPUCache())
  mapply(function(g, c, k, b) SpaNorm:::geneBlockCount(g, c, k, backend = backend, budget.bytes = b),
         grid$n_genes, grid$n_cells, grid$n_cov, grid$budget)
}
cnt_cuda_gpu <- count_for("cuda", "gpu")
cnt_cuda_auto <- count_for("cuda", "auto")
cnt_cuda_cpu <- count_for("cuda", "cpu")
cnt_mps_gpu <- count_for("mps", "gpu")
cnt_cpu_gpu <- count_for("cpu", "gpu")

idx_grid <- expand.grid(n_genes = c(1, 5, 10, 11, 100, 1001), nblocks = c(1, 2, 3, 4, 6, 7, 10))
idx_grid <- idx_grid[idx_grid$nblocks <= idx_grid$n_genes, ]
idx <- lapply(seq_len(nrow(idx_grid)), function(i) {
  b <- SpaNorm:::geneBlockIndices(idx_grid$n_genes[i], idx_grid$nblocks[i])
  # flat vector of gene indices (1-based) plus the block lengths
  list(flat = as.integer(unlist(b, use.names = FALSE)), lens = as.integer(lengths(b)))
})
names(idx) <- paste0("case", seq_along(idx))

# ---- NB distribution -------------------------------------------------------
g <- expand.grid(x = c(0, 1, 5, 50, 1000), mu = c(1e-8, 0.5, 3, 200),
                 size = c(0.01, 1, 10, 1e6, Inf))
g0 <- rbind(g, data.frame(x = c(0, 3), mu = 0, size = c(2, Inf)))  # mu = 0 point mass
pg <- expand.grid(p = c(0, 0.001, 0.1, 0.5, 0.9, 0.999, 1), mu = c(1e-8, 0.5, 3, 200),
                  size = c(0.01, 1, 10, 1e6, Inf))
pg <- rbind(pg, data.frame(p = c(0, 0.5, 1), mu = 0, size = c(2, 2, Inf)))
# round trip: p = pnbinom(q) sits exactly on a CDF step (tests R's fuzz)
rt <- g[is.finite(g$size) & g$x <= 50, ]
rt$p <- pnbinom(rt$x, mu = rt$mu, size = rt$size)

# matrix with a per-row size vector (recycled column-major by stats::)
set.seed(2)
xm <- matrix(rpois(4 * 6, 5), 4, 6)
mum <- matrix(rexp(4 * 6, 0.2), 4, 6)
szv <- c(0.5, 2, 20, Inf)

# ---- winsorisation ---------------------------------------------------------
set.seed(1)
alpha <- matrix(rnorm(60 * 4), 60, 4)
alpha[1, 1] <- 40; alpha[2, 3] <- -40
psi <- exp(rnorm(200, 0, 1.5))
psi[1] <- 1e6

write_ref(list(
  count = list(
    grid = list(n_genes = grid$n_genes, n_cells = grid$n_cells, n_cov = grid$n_cov, budget = grid$budget),
    cuda_gpu = as.integer(cnt_cuda_gpu), cuda_auto = as.integer(cnt_cuda_auto),
    cuda_cpu = as.integer(cnt_cuda_cpu), mps_gpu = as.integer(cnt_mps_gpu),
    cpu_gpu = as.integer(cnt_cpu_gpu)
  ),
  idx_grid = list(n_genes = idx_grid$n_genes, nblocks = idx_grid$nblocks),
  idx = idx,
  nb = list(
    x = g0$x, mu = g0$mu, size = g0$size,
    d = dnbinom(g0$x, mu = g0$mu, size = g0$size),
    dlog = dnbinom(g0$x, mu = g0$mu, size = g0$size, log = TRUE),
    p = pnbinom(g0$x, mu = g0$mu, size = g0$size),
    plog = pnbinom(g0$x, mu = g0$mu, size = g0$size, log.p = TRUE),
    qp = pg$p, qmu = pg$mu, qsize = pg$size,
    q = qnbinom(pg$p, mu = pg$mu, size = pg$size),
    rt_p = rt$p, rt_mu = rt$mu, rt_size = rt$size,
    rt_q = qnbinom(rt$p, mu = rt$mu, size = rt$size),
    xm = xm, mum = mum, szv = szv,
    dm = dnbinom(xm, mu = mum, size = szv),
    pm = pnbinom(xm, mu = mum, size = szv)
  ),
  win = list(
    alpha = alpha, psi = psi,
    cols4 = SpaNorm:::winsoriseCols(alpha, 4),
    cols1 = SpaNorm:::winsoriseCols(alpha, 1),
    psi4 = SpaNorm:::winsorisePsi(psi),
    psi1 = SpaNorm:::winsorisePsi(psi, winsor = 1)
  )
), "backend")
