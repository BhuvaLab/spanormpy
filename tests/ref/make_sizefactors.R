source("tests/ref/_common.R")
suppressPackageStartupMessages({
  library(SpatialExperiment)
  library(scran)
})

dump_case <- function(Y, extra = list()) {
  set.seed(1)
  cl <- as.integer(scran::quickCluster(Y))
  c(list(
    quick_cluster = cl,
    sf_given_clusters = as.numeric(scran::calculateSumFactors(Y, clusters = cl)),
    sf_no_clusters = as.numeric(scran::calculateSumFactors(Y)),
    fast = as.numeric(SpaNorm:::fastSizeFactors_intl(Y)),
    filter_0 = SpaNorm:::filterGenes_intl(Y, 0),
    filter_1 = SpaNorm:::filterGenes_intl(Y, 1),
    filter_0.1 = SpaNorm:::filterGenes_intl(Y, 0.1),
    filter_0.2 = SpaNorm:::filterGenes_intl(Y, 0.2),
    filter_0.5 = SpaNorm:::filterGenes_intl(Y, 0.5)
  ), extra)
}

# DLPFC subset (identical to make_dlpfc.R)
load("../SpaNorm/data/HumanDLPFC.rda")
spe <- get("HumanDLPFC")
keep <- filterGenes(spe, prop = 0.2)
spe <- spe[head(which(keep), 300), ]
Yd <- as.matrix(counts(spe))
dl <- dump_case(Yd)

# simulated data, plus extra cluster layouts exercising small / unequal clusters
sim <- sim_counts(200, 600, 4)
Ys <- sim$counts
xy <- sim$coords
manual <- as.integer(cut(xy[, 1], c(-Inf, 6, 14, Inf)))
manual[1:15] <- 4L # a cluster smaller than the smallest pool size (21)
sm <- dump_case(Ys, list(
  counts = Ys,
  manual_clusters = manual,
  sf_manual = as.numeric(scran::calculateSumFactors(Ys, clusters = manual)),
  sf_ref_clust = as.numeric(scran::calculateSumFactors(Ys, clusters = manual, ref.clust = 2L))
))

# a spot with zero total counts
Yz <- Ys
Yz[, 7] <- 0L
zero_msg <- tryCatch({
  scran::calculateSumFactors(Yz, clusters = rep(1L, ncol(Yz)))
  "no error"
}, error = function(e) conditionMessage(e))
qc_msg <- tryCatch({
  scran::quickCluster(Yz)
  "no error"
}, error = function(e) conditionMessage(e))

# heavy-tailed library sizes: gives non-positive pooled estimates (positive = TRUE branch,
# scuttle::cleanSizeFactors); located by a seed search (sd = 3, seed = 6)
set.seed(6)
ng <- 400; nc <- 66
lib <- exp(rnorm(nc, 0, 3))
mu <- outer(rexp(ng, 1 / (5 / mean(lib))), lib) * 20
Yl <- matrix(rpois(ng * nc, mu), ng, nc)
cll <- rep(1L, nc)
low_warn <- character()
lowsf <- withCallingHandlers(as.numeric(scran::calculateSumFactors(Yl, clusters = cll)),
  warning = function(w) { low_warn <<- c(low_warn, conditionMessage(w)); invokeRestart("muffleWarning") })
lowsf_nopos <- suppressWarnings(as.numeric(scran::calculateSumFactors(Yl, clusters = cll, positive = FALSE)))

write_ref(list(dl = dl, sim = sm, zero_msg = zero_msg, qc_zero_msg = qc_msg,
               low_counts = Yl, low_clusters = cll, low_sf = lowsf,
               low_sf_nopos = lowsf_nopos, low_warn = low_warn), "sizefactors")
