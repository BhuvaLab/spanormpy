# Shared helpers for reference generation. Scripts are run from the repo root.
suppressPackageStartupMessages({
  library(SpaNorm, lib.loc = "tests/ref/Rlib")
  library(rhdf5)
})

REF_DIR <- "tests/ref/data"
dir.create(REF_DIR, showWarnings = FALSE, recursive = TRUE)

`%||%` <- function(a, b) if (is.null(a)) b else a

.write_elem <- function(file, name, x) {
  if (is.list(x)) {
    h5createGroup(file, name)
    for (n in names(x)) .write_elem(file, paste0(name, "/", n), x[[n]])
    return(invisible())
  }
  lev <- NULL
  d <- dim(x) %||% length(x)
  if (is.factor(x)) {
    lev <- levels(x)
    x <- as.character(x)
  }
  if (is.logical(x)) {
    # logical -> int8
    h5createDataset(file, name, dims = d, storage.mode = "integer", size = 1)
    h5write(array(as.integer(x), dim = d), file, name)
  } else {
    h5write(array(x, dim = d), file, name)
  }
  if (!is.null(lev)) {
    fid <- H5Fopen(file)
    did <- H5Dopen(fid, name)
    h5writeAttribute(lev, did, "levels")
    H5Dclose(did)
    H5Fclose(fid)
  }
}

# Write a named list to tests/ref/data/<topic>.h5 (nested lists become groups,
# logical becomes int8, factors become character + a `levels` attribute).
write_ref <- function(x, topic) {
  file <- file.path(REF_DIR, paste0(topic, ".h5"))
  if (file.exists(file)) file.remove(file)
  h5createFile(file)
  for (n in names(x)) .write_elem(file, n, x[[n]])
  invisible(file)
}

# Deterministic NB simulation on a grid with spatial and library-size effects.
# Returns a genes x cells integer matrix plus coordinates (cells x 2).
sim_counts <- function(ngenes, ncells, seed) {
  set.seed(seed)
  side <- ceiling(sqrt(ncells))
  xy <- cbind(x = rep(seq_len(side), times = side), y = rep(seq_len(side), each = side))
  xy <- xy[seq_len(ncells), , drop = FALSE]
  libsize <- exp(rnorm(ncells, 0, 0.3) + 0.5 * scale(xy[, 1])[, 1])
  base <- rnorm(ngenes, 1.5, 0.8)
  spat <- matrix(rnorm(ngenes * 2, 0, 0.3), ngenes, 2)
  sxy <- scale(xy)
  logmu <- base + spat %*% t(sxy)
  mu <- exp(logmu) * rep(libsize, each = ngenes)
  size <- 5
  counts <- matrix(rnbinom(ngenes * ncells, mu = mu, size = size), ngenes, ncells)
  dimnames(counts) <- list(paste0("g", seq_len(ngenes)), paste0("c", seq_len(ncells)))
  list(counts = counts, coords = xy, libsize = libsize)
}
