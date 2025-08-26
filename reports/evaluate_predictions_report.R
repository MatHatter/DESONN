# ================================================================
# evaluate_predictions_report.R  (FULL FIX)
# ================================================================
source("utils/utils.R")

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(gridExtra)
  library(grid)
  library(pROC)
  library(PRROC)
  library(ggplotify)
  library(openxlsx)
})

EvaluatePredictionsReport <- function(
    X_validation, y_validation, probs,
    predicted_outputAndTime,
    threshold_function,           # kept for signature compatibility (not used)
    best_val_probs, best_val_labels,
    verbose = FALSE
) {
  # ------------------------- safety defaults -------------------------
  if (!exists("viewTables", inherits = TRUE)) viewTables <- FALSE
  if (!exists("ML_NN", inherits = TRUE))      ML_NN      <- FALSE
  if (!exists("train", inherits = TRUE))      train      <- FALSE
  
  # ------------------------- Inspect predictions/errors (optional) ---
  pred_vec   <- tryCatch(as.vector(predicted_outputAndTime$predicted_output_l2$learn_output),
                         error = function(e) rep(NA_real_, length.out = 0))
  err_vec    <- tryCatch(as.vector(predicted_outputAndTime$predicted_output_l2$error),
                         error = function(e) rep(NA_real_, length.out = 0))
  labels_vec <- tryCatch(as.vector(y_validation), error = function(e) rep(NA_real_, length.out = 0))
  max_points <- min(length(pred_vec), length(err_vec), length(labels_vec))
  
  if (max_points > 0) {
    plot(pred_vec[seq_len(max_points)], err_vec[seq_len(max_points)],
         main = "Prediction vs. Error", xlab = "Prediction", ylab = "Error",
         col = "steelblue", pch = 16)
    abline(h = 0, col = "gray", lty = 2)
    
    comment_error     <- character(max_points)
    comment_alignment <- character(max_points)
    for (i in seq_len(max_points)) {
      if (!is.na(labels_vec[i]) && labels_vec[i] == 1) {
        comment_error[i] <- paste0("Overpredicted by ", round(pred_vec[i] - 1, 3))
      } else {
        comment_error[i] <- paste0("Underpredicted by ", round(pred_vec[i], 3))
      }
      if (abs(pred_vec[i] - labels_vec[i]) > 0.5) {
        comment_alignment[i] <- paste0("Misaligned (", round(pred_vec[i], 2), " vs ", labels_vec[i], ")")
      } else {
        comment_alignment[i] <- "Aligned prediction"
      }
    }
    df_sample_commentary <- data.frame(
      Row = seq_len(max_points),
      Prediction = round(pred_vec[seq_len(max_points)], 5),
      Label = labels_vec[seq_len(max_points)],
      Error = round(err_vec[seq_len(max_points)], 5),
      Comment_Error = comment_error,
      Alignment_Comment = comment_alignment,
      stringsAsFactors = FALSE
    )
    if (isTRUE(viewTables)) View(df_sample_commentary)
  }
  
  # ------------------------- weights summary (robust) -----------------
  if (isTRUE(ML_NN)) {
    w_mat <- tryCatch(as.matrix(predicted_outputAndTime$weights_record[[1]]),
                      error = function(e) matrix(NA_real_, nrow = 0, ncol = 0))
    if (length(w_mat)) {
      weights_summary <- round(rowMeans(w_mat), 5)
      if (verbose) { cat(">> Multi-layer weights summary (first layer):\n"); print(weights_summary) }
    }
  } else {
    w_raw <- tryCatch(predicted_outputAndTime$weights_record[[1]], error = function(e) numeric(0))
    if (length(w_raw)) {
      weights_summary <- round(as.numeric(w_raw), 5)
      if (verbose) { cat(">> Single-layer weights summary:\n"); print(weights_summary) }
    }
  }
  
  # ------------------------- Tune on validation or current ------------
  if (isTRUE(train)) {
    probs_use  <- best_val_probs
    labels_use <- best_val_labels
  } else {
    probs_use  <- probs
    labels_use <- y_validation
  }
  
  # DO NOT flatten upfront; keep shapes to detect K correctly
  labels_mat <- as.matrix(labels_use)
  probs_mat  <- if (is.matrix(probs_use)) probs_use else matrix(probs_use, ncol = 1L)
  K <- max(ncol(labels_mat), ncol(probs_mat))
  
  # ------------------------- Threshold tuning -------------------------
  metric_for_tuning <- if (K == 1L) "accuracy" else "macro_f1"  # sensible default
  thr_res <- tune_threshold_accuracy(
    predicted_output = probs_mat,   # vector (binary) or matrix (multiclass)
    labels           = labels_mat,  # 0/1 or class ids, or one-hot matrix
    metric           = metric_for_tuning,
    threshold_grid             = seq(0.05, 0.95, by = 0.01),
    verbose          = FALSE
  )
  
  best_thresholds <- thr_res$thresholds   # length 1 (binary) or length K (multiclass)
  if (verbose) {
    if (length(best_thresholds) == 1L) {
      message(sprintf("Best threshold (%s-optimized): %.3f", metric_for_tuning, as.numeric(best_thresholds)))
    } else {
      message(sprintf("Best per-class thresholds (%s-optimized): %s",
                      metric_for_tuning, paste0(sprintf("%.3f", best_thresholds), collapse = ", ")))
    }
  }
  
  # ------------------------- Build predictions for accuracy() ---------
  heatmap_path <- NULL
  if (K == 1L) {
    # ===== Binary =====
    labels_flat <- as.integer(ifelse(as.vector(labels_mat) > 0, 1L, 0L))
    best_threshold <- as.numeric(best_thresholds)  # back-compat scalar
    
    binary_preds <- as.integer(as.vector(probs_mat) >= best_threshold)
    predicted_output_for_acc <- matrix(binary_preds, ncol = 1L)
    
    if (verbose) message(sprintf("Binary preds @ tuned threshold: mean=%0.3f", mean(binary_preds)))
    
    # accuracy() must be called BEFORE shadowing the name with a variable
    acc_prop <- accuracy(
      SONN = NULL, Rdata = NULL,
      labels = labels_mat,
      predicted_output = predicted_output_for_acc,
      verbose = FALSE
    )
    accuracy <- acc_prop                 # keep your variable names
    accuracy_percent <- accuracy * 100
    
    if (verbose) message(sprintf("Accuracy (binary @ tuned threshold): %.6f (%.3f%%)", accuracy, accuracy_percent))
    
    # Precision/Recall/F1 via your helper
    metrics <- tryCatch(evaluate_classification_metrics(binary_preds, labels_flat),
                        error = function(e) { if (verbose) message("metrics error: ", e$message); list() })
    
    # Confusion matrix (counts)
    TP <- sum(binary_preds == 1 & labels_flat == 1)
    TN <- sum(binary_preds == 0 & labels_flat == 0)
    FP <- sum(binary_preds == 1 & labels_flat == 0)
    FN <- sum(binary_preds == 0 & labels_flat == 1)
    
    conf_matrix_df <- data.frame(
      Actual    = c("0","0","1","1"),
      Predicted = c("0","1","0","1"),
      Count     = c(TN, FP, FN, TP)
    )
    if (verbose) { message("Confusion matrix (TN, FP, FN, TP): ", paste(c(TN, FP, FN, TP), collapse = ", ")); print(conf_matrix_df) }
    
    # Heatmap (binary)
    heatmap_path <- "confusion_matrix_heatmap.png"
    tryCatch({
      plot_conf_matrix <- ggplot(conf_matrix_df, aes(x = Predicted, y = Actual, fill = Count)) +
        geom_tile(color = "white") +
        geom_text(aes(label = Count), size = 6, fontface = "bold") +
        scale_fill_gradient(low = "white", high = "red") +
        labs(title = "Confusion Matrix Heatmap") +
        theme_minimal() +
        theme(plot.title = element_text(hjust = 0.5, face = "bold"))
      ggsave(heatmap_path, plot_conf_matrix, width = 5, height = 4, dpi = 300)
      while (!is.null(dev.list())) dev.off()
      if (verbose) print(plot_conf_matrix)
    }, error = function(e) message("❌ Failed to generate confusion matrix heatmap: ", e$message))
    
    # Combine features + labels + predictions (binary)
    Rdata_df <- as.data.frame(X_validation)
    combined_df <- cbind(
      Rdata_df,
      label = labels_flat,
      pred  = binary_preds,
      prob  = as.vector(probs_mat),
      threshold_used = best_threshold
    )
    
  } else {
    # ===== Multiclass =====
    best_threshold <- NA_real_          # no single scalar threshold
    pred_ids <- thr_res$y_pred_class    # class ids (1..K)
    
    # one-hot predictions for accuracy()
    N <- length(pred_ids)
    predicted_output_for_acc <- matrix(0, nrow = N, ncol = K)
    predicted_output_for_acc[cbind(seq_len(N), pred_ids)] <- 1
    
    # true class ids from labels (one-hot or class-id vector)
    if (ncol(labels_mat) > 1L) {
      y_true_ids <- max.col(labels_mat, ties.method = "first")
    } else {
      cls <- as.integer(labels_mat[, 1])
      if (min(cls, na.rm = TRUE) == 0L) cls <- cls + 1L
      cls[cls < 1L] <- 1L; cls[cls > K] <- K
      y_true_ids <- cls
    }
    
    acc_prop <- accuracy(
      SONN = NULL, Rdata = NULL,
      labels = labels_mat,
      predicted_output = predicted_output_for_acc,
      verbose = FALSE
    )
    accuracy <- acc_prop
    accuracy_percent <- accuracy * 100
    if (verbose) message(sprintf("Accuracy (multiclass @ tuned thresholds): %.6f (%.3f%%)", accuracy, accuracy_percent))
    
    # Confusion matrix (counts)
    conf_tab <- table(
      Actual    = factor(y_true_ids, levels = 1:K),
      Predicted = factor(pred_ids,   levels = 1:K)
    )
    conf_matrix_df <- as.data.frame(conf_tab); names(conf_matrix_df)[3] <- "Count"
    if (verbose) { message("Multiclass confusion matrix (counts):"); print(as.data.frame.matrix(conf_tab)) }
    
    # Heatmap (multiclass)
    heatmap_path <- "confusion_matrix_multiclass_heatmap.png"
    tryCatch({
      plot_conf_matrix_mc <- ggplot(conf_matrix_df,
                                    aes(x = factor(Predicted), y = factor(Actual), fill = Count)) +
        geom_tile(color = "white") +
        geom_text(aes(label = Count), size = 3, fontface = "bold") +
        scale_fill_gradient(low = "white", high = "red") +
        labs(title = "Confusion Matrix Heatmap (Multiclass)",
             x = "Predicted", y = "Actual") +
        theme_minimal() +
        theme(plot.title = element_text(hjust = 0.5, face = "bold"))
      ggsave(heatmap_path, plot_conf_matrix_mc, width = 6, height = 5, dpi = 300)
      while (!is.null(dev.list())) dev.off()
      if (verbose) print(plot_conf_matrix_mc)
    }, error = function(e) message("❌ Failed to generate multiclass heatmap: ", e$message))
    
    # Combine features + labels + predictions (multiclass)
    Rdata_df <- as.data.frame(X_validation)
    combined_df <- cbind(
      Rdata_df,
      label = y_true_ids,
      pred  = pred_ids
    )
    
    # No binary metrics in multiclass
    metrics <- NULL
  }
  
  # ------------------------- Probability-separation & calibration -----
  # (Binary only — skip for multiclass)
  if (K == 1L) {
    Rdata_with_labels <- cbind(as.data.frame(X_validation), Label = labels_flat)
    Rdata_predictions <- Rdata_with_labels %>%
      mutate(Predicted_Prob = as.vector(probs_mat),
             Predictions    = binary_preds)
    
    # Mean predicted probability by class
    prediction_means <- Rdata_predictions %>%
      group_by(Label = as.integer(Label)) %>%
      summarise(
        MeanProbability = mean(Predicted_Prob, na.rm = TRUE),
        StdDev          = sd(Predicted_Prob,   na.rm = TRUE),
        Count           = n(),
        .groups = "drop"
      )
    
    mean_0 <- prediction_means$MeanProbability[prediction_means$Label == 0]
    mean_1 <- prediction_means$MeanProbability[prediction_means$Label == 1]
    
    if (!is.na(mean_0) && !is.na(mean_1)) {
      commentary_text <- if (mean_0 < 0.2 && mean_1 > 0.8) {
        paste0("✅ Sharp separation: mean p(y=1|y=0)=", round(mean_0,4),
               " vs mean p(y=1|y=1)=", round(mean_1,4), ".")
      } else if (mean_0 > 0.35 && mean_1 < 0.65) {
        paste0("⚠️ Probabilities are close (", round(mean_0,4), " vs ", round(mean_1,4),
               ") — may need sharpening or more signal.")
      } else {
        paste0("ℹ️ Moderate separation (", round(mean_0,4), " vs ", round(mean_1,4), ").")
      }
    } else {
      commentary_text <- "❌ Mean probabilities NA — class imbalance or empty subset."
    }
    commentary_df_means <- data.frame(Interpretation = commentary_text)
    if (verbose) print(commentary_df_means)
    
    # ROC & PR curves (binary only)
    labels_numeric <- suppressWarnings(as.numeric(labels_flat))
    probs_numeric  <- suppressWarnings(as.numeric(as.vector(probs_mat)))
    
    # ROC
    tryCatch({
      if (length(unique(labels_numeric)) >= 2 && length(labels_numeric) == length(probs_numeric)) {
        roc_obj <- pROC::roc(response = labels_numeric, predictor = probs_numeric, quiet = TRUE)
        auc_value <- pROC::auc(roc_obj)
        auc_val_text <- paste("AUC =", round(as.numeric(auc_value), 4))
        if (verbose) cat("✅ AUC (ROC):", auc_val_text, "\n")
        roc_curve_plot <- ggplotify::as.ggplot(~{
          plot(roc_obj, col = "#1f77b4", lwd = 2, main = "ROC Curve")
          abline(a = 0, b = 1, lty = 2, col = "gray")
          text(0.6, 0.2, labels = auc_val_text, cex = 1.2)
        })
        ggsave("roc_curve.png", plot = roc_curve_plot, width = 6, height = 4, dpi = 300)
        while (!is.null(dev.list())) dev.off()
        if (verbose) print(roc_curve_plot)
      }
    }, error = function(e) message("⚠️ ROC block failed: ", e$message))
    
    # PR
    tryCatch({
      if (length(unique(labels_numeric)) >= 2 && length(labels_numeric) == length(probs_numeric)) {
        pr_obj <- PRROC::pr.curve(
          scores.class0 = probs_numeric[labels_numeric == 1],
          scores.class1 = probs_numeric[labels_numeric == 0],
          curve = TRUE
        )
        if (verbose) cat("✅ AUPRC:", round(pr_obj$auc.integral, 4), "\n")
        pr_curve_plot <- ggplotify::as.ggplot(~{
          plot(pr_obj, main = "Precision-Recall Curve", col = "#d62728", lwd = 2)
        })
        ggsave("pr_curve.png", plot = pr_curve_plot, width = 6, height = 4, dpi = 300)
        while (!is.null(dev.list())) dev.off()
        if (verbose) print(pr_curve_plot)
      }
    }, error = function(e) message("⚠️ PR block failed: ", e$message))
    
    # Calibration grouping (binary only)
    Rdata_predictions_cal <- Rdata_predictions %>%
      transmute(prob = as.numeric(Predicted_Prob),
                label = as.integer(Label)) %>%
      filter(!is.na(prob), !is.na(label)) %>%
      mutate(prob_bin = ntile(prob, 10)) %>%
      group_by(prob_bin) %>%
      summarise(
        bin_mid = mean(prob, na.rm = TRUE),
        actual_death_rate = mean(label, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      filter(is.finite(bin_mid), is.finite(actual_death_rate)) %>%
      mutate(prob_bin = factor(prob_bin))
    
    # Plots 1–3
    tryCatch({
      plot1 <- ggplot(Rdata_predictions_cal, aes(x = prob_bin, y = actual_death_rate)) +
        geom_col(fill = "steelblue") +
        labs(title = "Observed Rate by Risk Bin",
             x = "Predicted Risk Decile (1=low,10=high)",
             y = "Observed Positive Rate") +
        theme_minimal() +
        theme(plot.title = element_text(face = "bold", hjust = 0.5))
      ggsave("plot1_bar_actual_rate.png", plot1, width = 6, height = 4, dpi = 300)
      while (!is.null(dev.list())) dev.off()
      if (verbose) print(plot1)
    }, error = function(e) message("❌ Failed to generate plot1: ", e$message))
    
    tryCatch({
      plot2 <- ggplot(Rdata_predictions_cal, aes(x = bin_mid, y = actual_death_rate)) +
        geom_line(size = 1.2) +
        geom_point(size = 3) +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
        labs(title = "Calibration Curve",
             x = "Avg Predicted Probability",
             y = "Observed Rate") +
        theme_minimal() +
        theme(plot.title = element_text(face = "bold", hjust = 0.5))
      ggsave("plot2_calibration_curve.png", plot2, width = 6, height = 4, dpi = 300)
      while (!is.null(dev.list())) dev.off()
      if (verbose) print(plot2)
    }, error = function(e) message("❌ Failed to generate plot2: ", e$message))
    
    tryCatch({
      plot_overlay <- ggplot(Rdata_predictions_cal, aes(x = prob_bin)) +
        geom_col(aes(y = actual_death_rate, fill = "Observed Rate")) +
        geom_point(aes(y = bin_mid, color = "Avg Predicted Prob"),
                   size = 3, shape = 21, stroke = 1.2, fill = "white") +
        scale_fill_manual(values = c("Observed Rate" = "steelblue")) +
        scale_color_manual(values = c("Avg Predicted Prob" = "black")) +
        labs(title = "Overlay: Observed vs Predicted",
             x = "Predicted Risk Decile", y = "Rate", fill = NULL, color = NULL) +
        theme_minimal() +
        theme(legend.position = "bottom",
              plot.title = element_text(face = "bold", hjust = 0.5))
      ggsave("plot_overlay_with_legend_below.png", plot_overlay, width = 6, height = 4, dpi = 300)
      while (!is.null(dev.list())) dev.off()
      if (verbose) print(plot_overlay)
    }, error = function(e) message("❌ Failed to generate overlay plot: ", e$message))
    
    # Calibration table PNG
    tryCatch({
      calibration_table <- Rdata_predictions_cal %>%
        transmute(
          Bin = prob_bin,
          `Avg Predicted Prob` = round(bin_mid, 4),
          `Observed Rate`      = round(actual_death_rate, 4),
          `Difference (Obs - Pred)` = round(actual_death_rate - bin_mid, 4),
          `Calibration Quality` = case_when(
            abs(actual_death_rate - bin_mid) < 0.02 ~ "✅ Well Calibrated",
            actual_death_rate > bin_mid ~ "⚠️ Underconfident",
            TRUE ~ "⚠️ Overconfident"
          )
        )
      table_plot <- tableGrob(calibration_table, rows = NULL)
      ggsave("calibration_table.png", table_plot, width = 8, height = 4, dpi = 300)
      while (!is.null(dev.list())) dev.off()
    }, error = function(e) message("❌ Failed to generate calibration table PNG: ", e$message))
    
    # Misclassified samples (binary)
    wrong <- which(binary_preds != labels_flat)
    misclassified <- if (length(wrong)) {
      cbind(
        predicted_prob  = as.vector(probs_mat)[wrong],
        predicted_label = binary_preds[wrong],
        actual_label    = labels_flat[wrong],
        as.data.frame(X_validation)[wrong, , drop = FALSE]
      )
    } else {
      data.frame()
    }
    
    # Sorted misclassified + summary (robust to missing columns)
    misclassified_sorted <- if (nrow(misclassified) > 0) {
      misclassified %>%
        mutate(
          error = abs(predicted_prob - actual_label),
          Type  = ifelse(predicted_label == 1 & actual_label == 0, "False Positive", "False Negative")
        ) %>%
        arrange(desc(error))
    } else data.frame()
    
    summary_by_type <- tryCatch({
      if (nrow(misclassified_sorted) > 0) {
        wanted <- intersect(c("age","serum_creatinine","ejection_fraction","time"),
                            names(misclassified_sorted))
        if (length(wanted) == 0) {
          misclassified_sorted %>% group_by(Type) %>% summarise(n = n(), .groups = "drop")
        } else {
          misclassified_sorted %>%
            group_by(Type) %>%
            summarise(across(all_of(wanted), ~ mean(.x, na.rm = TRUE)), .groups = "drop")
        }
      } else data.frame()
    }, error = function(e) { if (verbose) message("summary_by_type error: ", e$message); data.frame() })
    
    if (nrow(summary_by_type) > 0) {
      saveRDS(summary_by_type, file = "summary_by_type.rds")
      if (verbose) cat("\n💾 summary_by_type saved as summary_by_type.rds\n")
    } else if (verbose) {
      message("⚠️ No misclassified samples to summarize.")
    }
    
  } else {
    # Multiclass: skip binary-specific artifacts
    prediction_means      <- data.frame()
    commentary_df_means   <- data.frame(Interpretation = "Calibration/PR/ROC skipped for multiclass.")
    misclassified         <- data.frame()
    misclassified_sorted  <- data.frame()
    summary_by_type       <- data.frame()
  }
  
  # ------------------------- Metrics summary table (for PNG) ----------
  if (K == 1L) {
    total    <- length(labels_flat)
    support0 <- sum(labels_flat == 0)
    support1 <- sum(labels_flat == 1)
    
    precision0 <- if ((TN + FN) > 0) TN / (TN + FN) else 0
    recall0    <- if ((TN + FP) > 0) TN / (TN + FP) else 0
    f1_0       <- if ((precision0 + recall0) > 0) 2 * precision0 * recall0 / (precision0 + recall0) else 0
    
    precision1 <- if ((TP + FP) > 0) TP / (TP + FP) else 0
    recall1    <- if ((TP + FN) > 0) TP / (TP + FN) else 0
    f1_1       <- if ((precision1 + recall1) > 0) 2 * precision1 * recall1 / (precision1 + recall1) else 0
    
    macro_precision <- mean(c(precision0, precision1))
    macro_recall    <- mean(c(recall0, recall1))
    macro_f1        <- mean(c(f1_0, f1_1))
    
    weighted_precision <- (support0 * precision0 + support1 * precision1) / max(total, 1)
    weighted_recall    <- (support0 * recall0 + support1 * recall1) / max(total, 1)
    weighted_f1        <- (support0 * f1_0 + support1 * f1_1) / max(total, 1)
    
    metrics_summary <- data.frame(
      Class      = c("0", "1", "macro avg", "weighted avg"),
      Precision  = c(precision0, precision1, macro_precision, weighted_precision),
      Recall     = c(recall0,    recall1,    macro_recall,    weighted_recall),
      F1_Score   = c(f1_0,       f1_1,       macro_f1,        weighted_f1),
      Support    = c(support0,   support1,   total,           total),
      Accuracy   = rep(accuracy, 4),
      TP         = rep(TP, 4),
      TN         = rep(TN, 4),
      FP         = rep(FP, 4),
      FN         = rep(FN, 4)
    )
  } else {
    # Build per-class metrics from confusion matrix for multiclass
    conf_mat <- xtabs(Count ~ Actual + Predicted, data = conf_matrix_df)  # KxK
    TPk <- diag(conf_mat)
    FPk <- colSums(conf_mat) - TPk
    FNk <- rowSums(conf_mat) - TPk
    Support <- rowSums(conf_mat)
    
    Prec_k <- ifelse((TPk + FPk) > 0, TPk / (TPk + FPk), 0)
    Rec_k  <- ifelse((TPk + FNk) > 0, TPk / (TPk + FNk), 0)
    F1_k   <- ifelse((Prec_k + Rec_k) > 0, 2 * Prec_k * Rec_k / (Prec_k + Rec_k), 0)
    
    macro_precision <- mean(Prec_k)
    macro_recall    <- mean(Rec_k)
    macro_f1        <- mean(F1_k)
    total           <- sum(Support)
    weighted_precision <- sum(Prec_k * Support) / max(total, 1)
    weighted_recall    <- sum(Rec_k  * Support) / max(total, 1)
    weighted_f1        <- sum(F1_k   * Support) / max(total, 1)
    
    metrics_summary <- rbind(
      data.frame(
        Class     = as.character(seq_len(K)),
        Precision = Prec_k,
        Recall    = Rec_k,
        F1_Score  = F1_k,
        Support   = Support,
        Accuracy  = rep(accuracy, K),
        TP        = TPk,
        TN        = NA_real_,
        FP        = FPk,
        FN        = FNk
      ),
      data.frame(
        Class     = c("macro avg", "weighted avg"),
        Precision = c(macro_precision, weighted_precision),
        Recall    = c(macro_recall,    weighted_recall),
        F1_Score  = c(macro_f1,        weighted_f1),
        Support   = c(total, total),
        Accuracy  = c(accuracy, accuracy),
        TP        = c(NA_real_, NA_real_),
        TN        = c(NA_real_, NA_real_),
        FP        = c(NA_real_, NA_real_),
        FN        = c(NA_real_, NA_real_)
      )
    )
  }
  
  # Save metrics table PNG (robust)
  tryCatch({
    png("metrics_summary.png", width = 1000, height = 400)
    grid.table(metrics_summary)
    dev.off()
  }, error = function(e) message("❌ Failed to save metrics_summary.png: ", e$message))
  
  # ------------------------- Metric commentary ------------------------
  commentary_df_metrics <- tryCatch({
    if (!exists("metrics") || is.null(metrics)) metrics <- list()
    f1   <- if (!is.null(metrics$F1))        metrics$F1        else NA
    prec <- if (!is.null(metrics$precision)) metrics$precision else NA
    rec  <- if (!is.null(metrics$recall))    metrics$recall    else NA
    
    txt <- c(paste("Accuracy:", round(accuracy, 4)))
    if (!is.na(f1))   txt <- c(txt, paste("F1 Score:", round(f1, 4)))
    if (!is.na(prec)) txt <- c(txt, paste("Precision:", round(prec, 4)))
    if (!is.na(rec))  txt <- c(txt, paste("Recall:", round(rec, 4)))
    
    if (is.finite(prec) && is.finite(rec)) {
      if (prec > rec + 0.05)   txt <- c(txt, "⚠️ Precision-dominant: conservative, may miss TPs.")
      else if (rec > prec + 0.05) txt <- c(txt, "⚠️ Recall-dominant: aggressive, may raise FPs.")
      else                        txt <- c(txt, "✅ Balanced precision and recall.")
    }
    data.frame(Interpretation = txt)
  }, error = function(e) { message("❌ Failed to generate commentary_df_metrics: ", e$message); data.frame(Interpretation = "⚠️ Metric interpretation unavailable.") })
  
  # ------------------------- Write to Excel ---------------------------
  wb <- createWorkbook()
  addWorksheet(wb, "Combined")
  suppressWarnings(writeData(wb, "Combined", combined_df))
  
  addWorksheet(wb, "Metrics_Summary")
  suppressWarnings(writeData(wb, "Metrics_Summary", metrics_summary))
  suppressWarnings(writeData(wb, "Metrics_Summary", commentary_df_metrics, startRow = nrow(metrics_summary) + 3))
  suppressWarnings(writeData(wb, "Metrics_Summary", conf_matrix_df, startRow = nrow(metrics_summary) + 6, rowNames = FALSE))
  
  if (!is.null(heatmap_path) && file.exists(heatmap_path)) {
    tryCatch(insertImage(wb, "Metrics_Summary", heatmap_path, startRow = nrow(metrics_summary) + 12, startCol = 1, width = 6, height = 4),
             error = function(e) message("⚠️ Could not insert heatmap: ", e$message))
  }
  
  # Optional: additional sheets if binary artifacts exist
  if (K == 1L) {
    addWorksheet(wb, "Prediction_Means")
    suppressWarnings(writeData(wb, "Prediction_Means", prediction_means))
    suppressWarnings(writeData(wb, "Prediction_Means", commentary_df_means, startRow = nrow(prediction_means) + 3))
    if (file.exists("plot1_bar_actual_rate.png"))
      tryCatch(insertImage(wb, "Prediction_Means", "plot1_bar_actual_rate.png", startRow = 20, startCol = 1, width = 6, height = 4), error = function(e) {})
    if (file.exists("plot2_calibration_curve.png"))
      tryCatch(insertImage(wb, "Prediction_Means", "plot2_calibration_curve.png", startRow = 34, startCol = 1, width = 6, height = 4), error = function(e) {})
    if (file.exists("plot_overlay_with_legend_below.png"))
      tryCatch(insertImage(wb, "Prediction_Means", "plot_overlay_with_legend_below.png", startRow = 48, startCol = 1, width = 6, height = 4), error = function(e) {})
    
    addWorksheet(wb, "Misclassified")
    suppressWarnings(writeData(wb, "Misclassified", misclassified_sorted))
  }
  
  saveWorkbook(wb, "Rdata_predictions.xlsx", overwrite = TRUE)
  
  # ------------------------- Debug lines ------------------------------
  if (length(best_thresholds) == 1L) {
    cat(sprintf("DEBUG >>> Final best_threshold (scalar): %.3f\n", as.numeric(best_thresholds)))
  } else {
    cat("DEBUG >>> Final best_thresholds (per-class): ",
        paste0(sprintf("%.3f", best_thresholds), collapse = ", "), "\n", sep = "")
  }
  cat("DEBUG >>> Tuned accuracy at best threshold(s):", round(accuracy, 5), "\n")
  
  # ------------------------- Return (back-compat + new) ---------------
  return(list(
    best_threshold   = best_threshold,       # scalar for binary, NA for multiclass
    best_thresholds  = best_thresholds,      # vector (len 1 for binary; K for multiclass)
    accuracy         = accuracy,
    accuracy_percent = accuracy * 100,
    metrics          = if (exists("metrics")) metrics else NULL,
    misclassified    = if (exists("misclassified")) misclassified else data.frame()
  ))
}
