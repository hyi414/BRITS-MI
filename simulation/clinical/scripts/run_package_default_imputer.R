args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7L) {
  stop(paste(
    "usage: run_package_default_imputer.R",
    "input.csv target_columns.txt method output.csv metadata.txt m seed"
  ))
}

input_path <- args[[1L]]
target_path <- args[[2L]]
method_name <- args[[3L]]
output_path <- args[[4L]]
metadata_path <- args[[5L]]
m <- as.integer(args[[6L]])
seed <- as.integer(args[[7L]])

dat <- read.csv(input_path, check.names = FALSE)
target_names <- readLines(target_path, warn = FALSE)
target_names <- target_names[nzchar(target_names)]
if (!all(target_names %in% names(dat))) {
  stop("Some requested target columns are absent from the input data.")
}

ignore <- NULL
if (".__ignore__" %in% names(dat)) {
  ignore <- as.logical(dat[[".__ignore__"]])
  dat[[".__ignore__"]] <- NULL
}

if (method_name %in% c("mice", "mice_selected")) {
  suppressPackageStartupMessages(library(mice))
  if (method_name == "mice_selected") {
    initial <- mice(dat, m = 1L, maxit = 0L, printFlag = FALSE)
    predictor_matrix <- initial$predictorMatrix
    predictor_matrix[,] <- 0L
    for (target in target_names) {
      parsed <- strcapture(
        "^y_t([0-9]+)_([0-9]+)$",
        target,
        proto = list(visit = integer(), marker = integer())
      )
      visit <- parsed$visit[[1L]]
      marker <- parsed$marker[[1L]]
      nearby_visits <- unique(pmax(1L, c(visit - 1L, visit, visit + 1L)))
      x_index <- (visit - 1L) * 3L + seq_len(3L)
      candidates <- unique(c(
        grep(paste0("^y_t[0-9]+_", marker, "$"), names(dat), value = TRUE),
        grep(paste0("^y_t", visit, "_[0-9]+$"), names(dat), value = TRUE),
        grep(paste0("^mask_t[0-9]+_", marker, "$"), names(dat), value = TRUE),
        grep(paste0("^mask_t", visit, "_[0-9]+$"), names(dat), value = TRUE),
        grep("^static_", names(dat), value = TRUE),
        grep("^image_feature_", names(dat), value = TRUE),
        grep("^image_stage_probability_", names(dat), value = TRUE),
        paste0("x_", x_index),
        paste0("visit_time_", nearby_visits),
        paste0("visit_gap_", nearby_visits),
        "observed_followup_length"
      ))
      candidates <- intersect(setdiff(candidates, target), names(dat))
      predictor_matrix[target, candidates] <- 1L
    }
    fit <- mice(
      dat,
      m = m,
      maxit = 5L,
      seed = seed,
      ignore = ignore,
      method = initial$method,
      predictorMatrix = predictor_matrix,
      printFlag = FALSE
    )
  } else {
    fit <- mice(
      dat,
      m = m,
      maxit = 5L,
      seed = seed,
      ignore = ignore,
      printFlag = FALSE
    )
  }
  completed <- complete(fit, action = "long", include = FALSE)
  output <- completed[, c(".imp", ".id", target_names), drop = FALSE]
  method_text <- paste(names(fit$method), fit$method, sep = "=")
  logged_events <- if (is.null(fit$loggedEvents)) 0L else nrow(fit$loggedEvents)
  metadata <- c(
    paste0("method=", method_name),
    paste0("package_version=", as.character(packageVersion("mice"))),
    paste0("m=", m),
    "maxit=5",
    "conditional_model=package default; predictive mean matching for incomplete continuous variables",
    paste0(
      "predictor_matrix=",
      if (method_name == "mice_selected")
        "prespecified same-marker history, same-visit biomarkers, masks, local timing, static covariates, and image context"
      else
        "package default with automatic constant/collinearity handling"
    ),
    paste0("ignore_rows=", if (is.null(ignore)) 0L else sum(ignore)),
    paste0("logged_events=", logged_events),
    paste0("methods=", paste(method_text, collapse = ";"))
  )
} else if (method_name == "missforest") {
  suppressPackageStartupMessages(library(missForest))
  set.seed(seed)
  fit <- missForest(dat, verbose = FALSE)
  output <- data.frame(.imp = 1L, .id = seq_len(nrow(dat)))
  output <- cbind(output, fit$ximp[, target_names, drop = FALSE])
    metadata <- c(
      "method=missforest",
      paste0("package_version=", as.character(packageVersion("missForest"))),
      "maxiter=10",
      "ntree=100",
      "mtry=floor(sqrt(number_of_predictors))",
      "nodesize=NULL (randomForest backend default)",
      "multiple_imputations=1",
    paste0("ignore_rows_not_supported=", if (is.null(ignore)) 0L else sum(ignore)),
    paste0("estimated_nrmse=", unname(fit$OOBerror[[1L]]))
  )
} else {
  stop("Unsupported method: ", method_name)
}

write.csv(output, output_path, row.names = FALSE)
writeLines(metadata, metadata_path)
