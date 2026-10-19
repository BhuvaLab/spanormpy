source("tests/ref/_common.R")
suppressPackageStartupMessages({
  library(SpatialExperiment)
  library(zellkonverter)
  library(SingleCellExperiment)
})
load("../SpaNorm/data/HumanDLPFC.rda")
spe <- get("HumanDLPFC")
keep <- filterGenes(spe, prop = 0.2)  # logical vector
spe <- spe[head(which(keep), 300), ]
# keep only counts and the AnnotatedCluster column; coordinates -> reducedDim "spatial"
spe_out <- SingleCellExperiment(
  assays = list(counts = as.matrix(counts(spe))),
  colData = S4Vectors::DataFrame(
    AnnotatedCluster = as.character(colData(spe)$AnnotatedCluster),
    row.names = colnames(spe)
  )
)
reducedDim(spe_out, "spatial") <- as.matrix(spatialCoords(spe))
writeH5AD(spe_out, "tests/ref/data/dlpfc_small.h5ad", X_name = "counts")
