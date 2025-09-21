# ===============================================================
# DeepDynamic — DDESONN
# Deep Dynamic Ensemble Self-Organizing Neural Network
# ---------------------------------------------------------------
# Copyright (c) 2024-2025 Mathew William Fok
# 
# Licensed for academic and personal research use only.
# Commercial use, redistribution, or incorporation into any
# profit-seeking product or service is strictly prohibited.
#
# This license applies to all versions of DeepDynamic/DDESONN,
# past, present, and future, including legacy releases.
#
# Intended future distribution: CRAN package.
# ===============================================================

# ===============================================================
# DDESONN — Summary (per-seed; uses fused as "test" for ensembles)
# ===============================================================

suppressPackageStartupMessages({ library(dplyr) })

summarize_stats <- function(v) c(
  count = sum(!is.na(v)),
  mean  = mean(v, na.rm = TRUE),
  std   = sd(v, na.rm = TRUE),
  min   = min(v, na.rm = TRUE),
  `25%` = quantile(v, 0.25, na.rm = TRUE, names = FALSE),
  `50%` = quantile(v, 0.50, na.rm = TRUE, names = FALSE),
  `75%` = quantile(v, 0.75, na.rm = TRUE, names = FALSE),
  max   = max(v, na.rm = TRUE)
)

# decide run root based on do_ensemble
is_ens <- isTRUE(get0("do_ensemble", inherits = TRUE, ifnotfound = TRUE))
root   <- file.path("artifacts", if (is_ens) "EnsembleRuns" else "SingleRuns")

# pick run dir: prefer .BM_DIR if it points under the root; else latest subfolder
RUN_DIR <- get0(".BM_DIR", inherits = TRUE, ifnotfound = NULL)
if (is.null(RUN_DIR) || !dir.exists(RUN_DIR) || !startsWith(normalizePath(RUN_DIR, winslash="/", mustWork=FALSE),
                                                            normalizePath(root, winslash="/", mustWork=FALSE))) {
  runs <- list.dirs(root, full.names = TRUE, recursive = FALSE)
  if (!length(runs)) stop("No run folders under: ", root)
  RUN_DIR <- runs[order(file.info(runs)$mtime, decreasing = TRUE)][1]
}
cat("[SUMMARY] Using RUN_DIR = ", RUN_DIR, "\n", sep = "")

# ---- load per-slot TRAIN table and reduce to per-seed ----
train_pat <- if (is_ens) "^Ensembles_Train_Acc_Val_Metrics_\\d+_seeds_.*\\.rds$"
else          "^SingleRun_Train_Acc_Val_Metrics_\\d+_seeds_.*\\.rds$"
train_file <- {
  fs <- list.files(RUN_DIR, pattern = train_pat, full.names = TRUE)
  if (!length(fs)) stop("Train/val metrics RDS not found in ", RUN_DIR)
  fs[order(file.info(fs)$mtime, decreasing = TRUE)][1]
}
tv <- readRDS(train_file)
names(tv) <- sub("^(performance_metric|relevance_metric)\\.", "", names(tv))
tv$seed <- coalesce(tv$seed, tv$SEED)

tv_seed <- tv %>%
  group_by(seed) %>%
  summarise(
    train_acc = suppressWarnings(max(best_train_acc, na.rm = TRUE)),
    val_acc   = suppressWarnings(max(best_val_acc,   na.rm = TRUE)),
    .groups = "drop"
  )

# ---- fused "test" metrics (ensemble only) ----
if (is_ens) {
  fdir <- file.path(RUN_DIR, "fused")
  ffiles <- list.files(fdir, pattern = "^fused_run\\d+_seed\\d+_.*\\.rds$", full.names = TRUE)
  if (!length(ffiles)) stop("No fused RDS files in ", fdir)
  
  fused_rows <- do.call(rbind, lapply(ffiles, function(f) {
    z <- readRDS(f)
    m <- z$metrics
    # add seed/run from filename
    bn <- basename(f)
    m$seed      <- as.integer(sub(".*_seed(\\d+)_.*", "\\1", bn, perl = TRUE))
    m$run_index <- as.integer(sub("^fused_run(\\d+).*", "\\1", bn, perl = TRUE))
    m
  }))
  
  # pick one method to represent the ensemble test (change if desired)
  method_pick <- "Ensemble_wavg"  # or "Ensemble_avg", "Ensemble_vote_soft", "Ensemble_vote_hard"
  fused_best <- fused_rows %>% filter(kind == method_pick)
  
  fused_seed <- fused_best %>%
    select(seed, accuracy, precision, recall, f1) %>%
    rename(test_acc = accuracy, test_precision = precision, test_recall = recall, test_f1 = f1)
} else {
  fused_seed <- tv_seed %>% mutate(test_acc = NA_real_, test_precision = NA_real_, test_recall = NA_real_, test_f1 = NA_real_)
}

# ---- merge and summarise ----
merged <- inner_join(tv_seed, fused_seed %>% select(seed, test_acc, test_precision, test_recall, test_f1), by = "seed") %>%
  arrange(seed)

summary_all <- sapply(merged[c("seed","train_acc","val_acc","test_acc")], summarize_stats)
summary_all <- round(as.data.frame(summary_all), 4)

cat("=== Summary (per seed; fused used as test when ensemble) ===\n")
print(summary_all)

cat("\n=== Per-seed table ===\n")
print(merged)
