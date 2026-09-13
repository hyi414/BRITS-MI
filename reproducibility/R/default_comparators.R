# Package-default wide-format comparators used for reproducibility checks.

run_mice_default <- function(data, m = 50L, seed = 1L) {
  if (!requireNamespace("mice", quietly = TRUE)) {
    stop("Install the 'mice' package before running this comparator.")
  }
  mice::mice(
    data,
    m = m,
    maxit = 5L,
    seed = seed,
    printFlag = FALSE
  )
}

run_missforest_default <- function(data, m = 5L, seed = 1L) {
  if (!requireNamespace("missForest", quietly = TRUE)) {
    stop("Install the 'missForest' package before running this comparator.")
  }
  lapply(seq_len(m), function(draw) {
    set.seed(seed + draw - 1L)
    missForest::missForest(
      data,
      maxiter = 10L,
      ntree = 100L,
      verbose = FALSE
    )$ximp
  })
}

# The caller must put every method on the same subject split, wide-format
# variable set, outcome-information rule, and artificial mask. Complete each
# MICE dataset with mice::complete(), then refit and pool the downstream model.
# Repeated missForest fits quantify algorithmic variation but are not guaranteed
# to be proper multiple imputations; report that limitation explicitly.
