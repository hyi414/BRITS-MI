args <- commandArgs(trailingOnly=TRUE)
directory <- args[[1]]
method <- args[[2]]
m <- as.integer(args[[3]])
seed <- as.integer(args[[4]])
maxit <- as.integer(args[[5]])
p <- as.integer(args[[6]])
dat <- read.csv(file.path(directory, "input.csv"), check.names=FALSE)
if ("outcome" %in% names(dat)) dat$outcome <- factor(dat$outcome)
targets <- names(dat)[seq_len(p)]
set.seed(seed)
if (method == "mice") {
  suppressPackageStartupMessages(library(mice))
  fit <- mice(dat, m=m, maxit=maxit, seed=seed, printFlag=FALSE)
  draws <- lapply(seq_len(m), function(k) {
    out <- complete(fit, k)[, targets, drop=FALSE]
    out$imputation <- k
    out
  })
  events <- fit$loggedEvents
  if (!is.null(events)) write.csv(events, file.path(directory, "events.csv"), row.names=FALSE)
  target_models <- data.frame(target=targets, conditional_method=fit$method[targets],
    n_predictors=rowSums(fit$predictorMatrix[targets, , drop=FALSE] != 0),
    outcome_predictor=if("outcome" %in% colnames(fit$predictorMatrix)) fit$predictorMatrix[targets, "outcome"] != 0 else FALSE,
    image_predictors=rowSums(fit$predictorMatrix[targets, grepl("^image_", colnames(fit$predictorMatrix)), drop=FALSE] != 0))
  write.csv(target_models, file.path(directory, "target_models.csv"), row.names=FALSE)
  audit <- data.frame(method="mice", version=as.character(packageVersion("mice")),
                      m=m, maxit=maxit, logged_events=if(is.null(events)) 0L else nrow(events))
} else if (method == "missforest") {
  suppressPackageStartupMessages(library(missForest))
  fit <- missForest(dat, verbose=FALSE)
  out <- as.data.frame(fit$ximp[, targets, drop=FALSE])
  out$imputation <- 1L
  draws <- list(out)
  audit <- data.frame(method="missForest", version=as.character(packageVersion("missForest")),
                      m=1L, maxit=10L, logged_events=0L)
} else stop("unknown method")
write.csv(do.call(rbind, draws), file.path(directory, "draws.csv"), row.names=FALSE)
write.csv(audit, file.path(directory, "audit.csv"), row.names=FALSE)
