# --- Summarize helper ---
summarize_stats <- function(df) {
  c(
    count = length(df),
    mean  = mean(df, na.rm = TRUE),
    std   = sd(df, na.rm = TRUE),
    min   = min(df, na.rm = TRUE),
    `25%` = quantile(df, 0.25, na.rm = TRUE, names = FALSE),
    `50%` = quantile(df, 0.50, na.rm = TRUE, names = FALSE),
    `75%` = quantile(df, 0.75, na.rm = TRUE, names = FALSE),
    max   = max(df, na.rm = TRUE)
  )
}

# --- Merge train/val/test as before ---
merged <- merge(
  train_acc_validation_metrics_runs_20250919_172310_50_seeds[, c("seed","best_train_acc","accuracy")],
  metrics_test_20250919_162655_50_seeds[, c("seed","accuracy")],
  by = "seed",
  suffixes = c("_val","_test")
)
colnames(merged) <- c("seed","train_acc","val_acc","test_acc")

# --- Compute and transpose ---
summary_all <- sapply(
  merged[c("seed","train_acc","val_acc","test_acc")],
  summarize_stats
)

# Flip orientation
summary_all <- as.data.frame(summary_all)
summary_all <- round(summary_all, 4)

print("=== Summary ===")
print(summary_all)
