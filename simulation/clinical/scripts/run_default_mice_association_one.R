args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) {
  stop("usage: run_default_mice_association_one.R input.csv output.csv metadata.txt m seed")
}

input_path <- args[[1L]]
output_path <- args[[2L]]
metadata_path <- args[[3L]]
m <- as.integer(args[[4L]])
seed <- as.integer(args[[5L]])

suppressPackageStartupMessages(library(mice))

dat <- read.csv(input_path, check.names = FALSE)
target_names <- c(
  "ast_v4", "alt_v4", "platelet_v4",
  "ast_v5", "alt_v5", "platelet_v5",
  "ast_v6", "alt_v6", "platelet_v6"
)

# Keep the default MICE conditional models, predictor matrix, visit sequence,
# and five iterations. Only m is set to 20 to match the poster comparator.
fit <- mice(
  dat,
  m = m,
  maxit = 5L,
  seed = seed,
  printFlag = FALSE
)

completed <- complete(fit, action = "long", include = FALSE)
write.csv(
  completed[, c(".imp", ".id", target_names)],
  output_path,
  row.names = FALSE
)

method_text <- paste(names(fit$method), fit$method, sep = "=")
logged_events <- if (is.null(fit$loggedEvents)) 0L else nrow(fit$loggedEvents)
writeLines(
  c(
    paste0("mice_version=", as.character(packageVersion("mice"))),
    paste0("m=", m),
    "maxit=5",
    paste0("methods=", paste(method_text, collapse = ";")),
    paste0("logged_events=", logged_events)
  ),
  metadata_path
)
