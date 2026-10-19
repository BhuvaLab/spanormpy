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

# full data for the GPU benchmark (all spots, all genes passing filterGenes(prop = 0.2));
# git-ignored, not used by the tests
spe_full <- get("HumanDLPFC")
spe_full <- spe_full[which(filterGenes(spe_full, prop = 0.2)), ]
full_out <- SingleCellExperiment(
  assays = list(counts = as.matrix(counts(spe_full))),
  colData = S4Vectors::DataFrame(
    AnnotatedCluster = as.character(colData(spe_full)$AnnotatedCluster),
    row.names = colnames(spe_full)
  )
)
reducedDim(full_out, "spatial") <- as.matrix(spatialCoords(spe_full))
writeH5AD(full_out, "tests/ref/data/dlpfc_full.h5ad", X_name = "counts")
