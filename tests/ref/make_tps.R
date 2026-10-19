source("tests/ref/_common.R")
suppressPackageStartupMessages(library(SingleCellExperiment))
spe <- zellkonverter::readH5AD("tests/ref/data/dlpfc_small.h5ad")
xy <- reducedDim(spe, "spatial")
x <- as.numeric(xy[, 1]); y <- as.numeric(xy[, 2])
out <- list(x = x, y = y)
strip <- function(m) { a <- attr(m, "df.tps"); attributes(m) <- list(dim = dim(m)); list(B = m, df = as.integer(a)) }
for (df in c(2, 6, 10)) {
  for (ce in c(TRUE, FALSE)) {
    r <- tpsBasis(x, y, df = df, center = ce)
    attr(r, "scaled:center") <- NULL
    s <- strip(r)
    out[[sprintf("tps_df%d_c%d", df, ce)]] <- s
  }
}
out$tps_pair <- strip(tpsBasis(x, y, df = c(4, 3)))
# reference subset
sub <- which(x < quantile(x, 0.4) & y > quantile(y, 0.3))
out$sub_idx <- sub
r <- tpsBasis(x[sub], y[sub], df = 6, ref.x = x, ref.y = y)
cm <- attr(r, "scaled:center"); s <- strip(r); s$center <- as.numeric(cm); out$tps_ref <- s
# extrapolation
ex <- tpsBasis(c(min(x) - 500, max(x) + 800, mean(x)), c(max(y) + 300, min(y) - 100, mean(y)),
               df = 6, ref.x = x, ref.y = y)
out$extrap <- strip(ex)
# bs.tps on SpaNorm-scaled coordinates
sx <- (x - min(x)) / (max(x) - min(x)) - 0.5
sy <- (y - min(y)) / (max(y) - min(y)) - 0.5
for (df in c(1, 3, 6, 10)) out[[sprintf("bs_df%d", df)]] <- strip(SpaNorm:::bs.tps(sx, sy, df))
out$bs_raw6 <- strip(SpaNorm:::bs.tps(x, y, 6))
write_ref(out, "tps")
