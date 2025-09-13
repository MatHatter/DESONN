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
    X_validation, y_validation, CLASSIFICATION_MODE,
    probs,
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
  
  # ------------------------- Select data for evaluation ----------------
  if (isTRUE(train)) {
    probs_use  <- best_val_probs
    labels_use <- best_val_labels
  } else {
    probs_use  <- probs
    labels_use <- y_validation
  }
  
  # Shapes (do not flatten yet)
  labels_mat <- as.matrix(labels_use)
  probs_mat  <- if (is.matrix(probs_use)) probs_use else matrix(probs_use, ncol = 1L)
  
  # If labels are a single non-numeric column (e.g., "A","B","C"), coerce to one-hot
  need_one_hot <- (ncol(labels_mat) == 1L) &&
    all(is.na(suppressWarnings(as.numeric(labels_mat[, 1]))))
  if (need_one_hot) {
    f <- factor(labels_mat[, 1], levels = sort(unique(labels_mat[, 1])))
    y_ids <- as.integer(f)                          # 1..K
    K_from_labels <- length(levels(f))
    oh <- matrix(0L, nrow = length(y_ids), ncol = K_from_labels)
    oh[cbind(seq_along(y_ids), y_ids)] <- 1L
    colnames(oh) <- levels(f)
    labels_mat <- oh
    if (verbose) {
      cat("[INFO] Coerced character labels to one-hot.\n")
      cat("[INFO] Class level → column mapping:\n")
      print(setNames(seq_len(K_from_labels), levels(f)))
    }
  }
  
  # Align rows (trim only)
  N_eff <- min(nrow(probs_mat), nrow(labels_mat))
  if (N_eff <= 0) stop("No overlapping rows between probs and labels.")
  probs_mat  <- probs_mat [seq_len(N_eff), , drop = FALSE]
  labels_mat <- labels_mat[seq_len(N_eff), , drop = FALSE]
  
  # ========================= REGRESSION BRANCH (NEW) =========================
  if (tolower(CLASSIFICATION_MODE) %in% c("regression", "reg")) {
    # Treat probs_mat as continuous predictions; labels_mat as true y
    y  <- suppressWarnings(as.numeric(labels_mat[, 1]))
    yhat <- suppressWarnings(as.numeric(probs_mat[, 1]))
    keep <- is.finite(y) & is.finite(yhat)
    y    <- y[keep]
    yhat <- yhat[keep]
    
    if (length(y) == 0L) stop("Regression mode: no finite overlapping y / yhat.")
    
    residuals <- yhat - y
    SSE <- sum((yhat - y)^2)
    SST <- sum((y - mean(y))^2)
    RMSE <- sqrt(mean(residuals^2))
    MAE  <- mean(abs(residuals))
    MAPE <- if (any(y != 0)) mean(abs(residuals / y)) else NA_real_
    R2   <- if (SST > 0) 1 - SSE / SST else NA_real_
    Corr <- suppressWarnings(stats::cor(y, yhat))
    
    # Plots (saved to PNG)
    # 1) Scatter y vs yhat
    tryCatch({
      df_sc <- data.frame(y = y, yhat = yhat)
      p_sc <- ggplot(df_sc, aes(x = y, y = yhat)) +
        geom_point(alpha = 0.6) +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
        labs(title = "Regression: y (actual) vs ŷ (predicted)",
             x = "Actual", y = "Predicted") +
        theme_minimal()
      ggsave("reg_scatter_y_vs_yhat.png", p_sc, width = 6, height = 4, dpi = 300)
      while (!is.null(dev.list())) dev.off()
      if (verbose) print(p_sc)
    }, error = function(e) message("⚠️ Scatter plot failed: ", e$message))
    
    # 2) Residuals vs Fitted
    tryCatch({
      df_rf <- data.frame(fitted = yhat, resid = residuals)
      p_rf <- ggplot(df_rf, aes(x = fitted, y = resid)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
        geom_point(alpha = 0.6) +
        labs(title = "Residuals vs Fitted", x = "Fitted (ŷ)", y = "Residual (ŷ - y)") +
        theme_minimal()
      ggsave("reg_residuals_vs_fitted.png", p_rf, width = 6, height = 4, dpi = 300)
      while (!is.null(dev.list())) dev.off()
      if (verbose) print(p_rf)
    }, error = function(e) message("⚠️ Residuals vs Fitted failed: ", e$message))
    
    # 3) Residuals histogram
    tryCatch({
      df_rh <- data.frame(resid = residuals)
      p_rh <- ggplot(df_rh, aes(x = resid)) +
        geom_histogram(bins = 30) +
        labs(title = "Residuals Histogram", x = "Residual", y = "Count") +
        theme_minimal()
      ggsave("reg_residuals_hist.png", p_rh, width = 6, height = 4, dpi = 300)
      while (!is.null(dev.list())) dev.off()
      if (verbose) print(p_rh)
    }, error = function(e) message("⚠️ Residuals histogram failed: ", e$message))
    
    # Combined DF
    Rdata_df <- as.data.frame(X_validation)
    combined_df <- cbind(
      Rdata_df[keep, , drop = FALSE],
      label = y,
      pred  = yhat,
      residual = residuals
    )
    
    # Metrics summary (regression)
    metrics_summary <- data.frame(
      Metric = c("RMSE", "MAE", "MAPE", "R2", "Correlation"),
      Value  = c(RMSE, MAE, MAPE, R2, Corr)
    )
    
    # Commentary
    commentary_df_metrics <- data.frame(
      Interpretation = c(
        paste("RMSE:", round(RMSE, 6)),
        paste("MAE :", round(MAE,  6)),
        paste("R²  :", ifelse(is.finite(R2), round(R2, 6), NA)),
        paste("Corr:", ifelse(is.finite(Corr), round(Corr, 6), NA))
      )
    )
    
    # Excel
    wb <- createWorkbook()
    addWorksheet(wb, "Combined")
    suppressWarnings(writeData(wb, "Combined", combined_df))
    addWorksheet(wb, "Metrics_Summary")
    suppressWarnings(writeData(wb, "Metrics_Summary", metrics_summary))
    suppressWarnings(writeData(wb, "Metrics_Summary", commentary_df_metrics, startRow = nrow(metrics_summary) + 3))
    # Insert plots if present
    if (file.exists("reg_scatter_y_vs_yhat.png")) {
      tryCatch(insertImage(wb, "Metrics_Summary", "reg_scatter_y_vs_yhat.png", startRow = 20, startCol = 1, width = 6, height = 4), error = function(e) {})
    }
    if (file.exists("reg_residuals_vs_fitted.png")) {
      tryCatch(insertImage(wb, "Metrics_Summary", "reg_residuals_vs_fitted.png", startRow = 35, startCol = 1, width = 6, height = 4), error = function(e) {})
    }
    if (file.exists("reg_residuals_hist.png")) {
      tryCatch(insertImage(wb, "Metrics_Summary", "reg_residuals_hist.png", startRow = 50, startCol = 1, width = 6, height = 4), error = function(e) {})
    }
    saveWorkbook(wb, "Rdata_predictions.xlsx", overwrite = TRUE)
    
    # Return (keep contract; accuracy not applicable)
    return(list(
      best_threshold   = NA_real_,
      best_thresholds  = NA_real_,
      accuracy         = NA_real_,
      accuracy_percent = NA_real_,
      metrics          = list(RMSE = RMSE, MAE = MAE, MAPE = MAPE, R2 = R2, Correlation = Corr),
      misclassified    = data.frame()
    ))
  }
  # ======================= END REGRESSION BRANCH (NEW) =======================
  
  # Infer binary vs multiclass using helpers from utils
  inf <- infer_is_binary(labels_mat, probs_mat)
  is_binary <- isTRUE(inf$is_binary)
  K <- max(1L, ncol(labels_mat), ncol(probs_mat))
  
  # ------------------------- DEBUG DIAGNOSTICS -------------------------
  cat("\n[DBG] --- threshold tuning inputs ---\n")
  
  dbg_shape <- function(x) {
    nr <- if (!is.null(dim(x))) nrow(x) else length(x)
    nc <- if (!is.null(dim(x))) ncol(x) else 1L
    tp <- paste(class(x), collapse = "/")
    c(nrow = nr, ncol = nc, type = tp)
  }
  
  dbg_count_nonfinite <- function(x) {
    if (is.null(dim(x))) x <- matrix(x, ncol = 1L)
    nf  <- sum(!is.finite(x))
    na_ <- sum(is.na(x))
    nan <- sum(is.nan(x))
    infv <- sum(is.infinite(x))
    setNames(c(nf, na_, nan, infv), c("nonfinite", "NA", "NaN", "Inf"))
  }
  
  if(verbose){
    cat("[DBG] probs_mat shape/type: ", paste(dbg_shape(probs_mat), collapse = " | "), "\n", sep = "")
    cat("[DBG] labels_mat shape/type: ", paste(dbg_shape(labels_mat), collapse = " | "), "\n", sep = "")
    cat("[DBG] probs non-finite: ", paste(dbg_count_nonfinite(probs_mat), collapse = " | "), "\n", sep = "")
    cat("[DBG] labels non-finite: ", paste(dbg_count_nonfinite(labels_mat), collapse = " | "), "\n", sep = "")
    
    head_n <- min(5L, nrow(probs_mat))
    cat("[DBG] head(probs_mat):\n"); print(utils::head(as.data.frame(probs_mat), head_n))
    cat("[DBG] head(labels_mat):\n"); print(utils::head(as.data.frame(labels_mat), head_n))
    
    cat("[DBG] inferred is_binary: ", is_binary, " | K: ", K, " (1=binary, >1=multiclass)\n", sep = "")
    cat("[DBG] --- end threshold tuning inputs ---\n\n")
  }
  # ------------------------- Build predictions & metrics --------------
  heatmap_path <- NULL
  metrics <- NULL
  best_thresholds <- NA_real_
  best_threshold  <- NA_real_
  
  if (is_binary) {
    # Labels → 0/1 via util; Predictions → positive-class prob via util
    labels_flat <- labels_to_binary_vec(labels_mat)
    p_pos       <- preds_to_pos_prob(probs_mat)
    
    if (is.null(labels_flat) || is.null(p_pos)) {
      stop("Binary inference succeeded, but could not derive binary labels/probabilities.")
    }
    
    # Tune threshold safely (use util to sanitize grid)
    metric_for_tuning <- "accuracy"
    thr_grid <- sanitize_grid(seq(0.05, 0.95, by = 0.01))
    
    thr_res <- tryCatch(
      tune_threshold_accuracy(
        predicted_output = matrix(p_pos, ncol = 1L),
        labels           = matrix(labels_flat, ncol = 1L),
        metric           = metric_for_tuning,
        threshold_grid   = thr_grid,
        verbose          = FALSE
      ),
      error = function(e) {
        message("tune_threshold_accuracy failed: ", e$message, " — using 0.5.")
        list(thresholds = 0.5, y_pred_class = NULL)
      }
    )
    
    best_thresholds <- thr_res$thresholds
    best_threshold  <- as.numeric(best_thresholds)
    
    binary_preds <- as.integer(p_pos >= best_threshold)
    predicted_output_for_acc <- matrix(binary_preds, ncol = 1L)
    
    if (verbose) message(sprintf("Binary preds @ tuned threshold: mean=%0.3f", mean(binary_preds)))
    
    acc_prop <- accuracy(SONN, Rdata, labels = matrix(labels_flat, ncol = 1L),
                         CLASSIFICATION_MODE,
                         predicted_output = predicted_output_for_acc,
                         verbose = FALSE
    )
    accuracy <- acc_prop
    accuracy_percent <- accuracy * 100
    if (verbose) message(sprintf("Accuracy (binary @ tuned threshold): %.6f (%.3f%%)", accuracy, accuracy_percent))
    
    # Precision/Recall/F1 via helper
    metrics <- safe_eval_metrics(binary_preds, labels_flat, verbose = verbose)
    
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
    if (verbose) {
      message("Confusion matrix (TN, FP, FN, TP): ", paste(c(TN, FP, FN, TP), collapse = ", "))
      print(conf_matrix_df)
    }
    
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
    
    # Combined DF (binary)
    Rdata_df <- as.data.frame(X_validation)
    combined_df <- cbind(
      Rdata_df,
      label = labels_flat,
      pred  = binary_preds,
      prob  = p_pos,
      threshold_used = best_threshold
    )
    
  } else {
    # ===== Multiclass: no threshold tuning — use argmax =====
    pred_ids <- max.col(probs_mat, ties.method = "first")
    
    # True class ids
    if (ncol(labels_mat) > 1L) {
      y_true_ids <- max.col(labels_mat, ties.method = "first")
    } else {
      cls <- suppressWarnings(as.integer(labels_mat[, 1]))
      if (min(cls, na.rm = TRUE) == 0L) cls <- cls + 1L
      cls[!is.finite(cls)] <- 1L
      cls[cls < 1L] <- 1L; cls[cls > K] <- K
      y_true_ids <- cls
    }
    
    # Build one-hot predictions for accuracy() using util
    predicted_output_for_acc <- one_hot_from_ids(pred_ids, K, N_eff)
    
    acc_prop <- accuracy(
      SONN = NULL, Rdata = NULL,
      labels = labels_mat,
      CLASSIFICATION_MODE = CLASSIFICATION_MODE,
      predicted_output = predicted_output_for_acc,
      verbose = FALSE
    )
    accuracy <- acc_prop
    accuracy_percent <- accuracy * 100
    if (verbose) message(sprintf("Accuracy (multiclass, argmax): %.6f (%.3f%%)", accuracy, accuracy_percent))
    
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
    
    # Combined DF (multiclass)
    Rdata_df <- as.data.frame(X_validation)
    combined_df <- cbind(
      Rdata_df,
      label = y_true_ids,
      pred  = pred_ids
    )
  }
  
  # ------------------------- Probability separation / calibration -----
  # (Binary only)
  if (is_binary) {
    Rdata_with_labels <- cbind(as.data.frame(X_validation), Label = labels_flat)
    Rdata_predictions <- Rdata_with_labels %>%
      mutate(Predicted_Prob = as.numeric(p_pos),
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
    probs_numeric  <- suppressWarnings(as.numeric(p_pos))
    
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
    
    # Misclassified samples (binary)
    wrong <- which(binary_preds != labels_flat)
    misclassified <- if (length(wrong)) {
      cbind(
        predicted_prob  = p_pos[wrong],
        predicted_label = binary_preds[wrong],
        actual_label    = labels_flat[wrong],
        as.data.frame(X_validation)[wrong, , drop = FALSE]
      )
    } else {
      data.frame()
    }
    
    # Sorted misclassified + summary
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
  if (is_binary) {
    total    <- length(labels_flat)
    support0 <- sum(labels_flat == 0)
    support1 <- sum(labels_flat == 1)
    
    # If TP/TN/FP/FN not defined (should be, but guard)
    if (!exists("TP")) TP <- sum(binary_preds == 1 & labels_flat == 1)
    if (!exists("TN")) TN <- sum(binary_preds == 0 & labels_flat == 0)
    if (!exists("FP")) FP <- sum(binary_preds == 1 & labels_flat == 0)
    if (!exists("FN")) FN <- sum(binary_preds == 0 & labels_flat == 1)
    
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
      TP         = c(TN, TP, NA_real_, NA_real_),  # keep TN/TP visible in first two rows
      TN         = c(TN, TP, NA_real_, NA_real_),  # (note: columns kept for compatibility)
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
      if (prec > rec + 0.05)      txt <- c(txt, "⚠️ Precision-dominant: conservative, may miss TPs.")
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
  if (exists("conf_matrix_df")) {
    suppressWarnings(writeData(wb, "Metrics_Summary", conf_matrix_df, startRow = nrow(metrics_summary) + 6, rowNames = FALSE))
  }
  
  if (!is.null(heatmap_path) && file.exists(heatmap_path)) {
    tryCatch(insertImage(wb, "Metrics_Summary", heatmap_path, startRow = nrow(metrics_summary) + 12, startCol = 1, width = 6, height = 4),
             error = function(e) message("⚠️ Could not insert heatmap: ", e$message))
  }
  
  # Optional: additional sheets if binary artifacts exist
  if (is_binary) {
    addWorksheet(wb, "Prediction_Means")
    if (exists("prediction_means")) suppressWarnings(writeData(wb, "Prediction_Means", prediction_means))
    if (exists("commentary_df_means")) suppressWarnings(writeData(wb, "Prediction_Means", commentary_df_means, startRow = if (exists("prediction_means")) nrow(prediction_means) + 3 else 3))
    if (file.exists("plot1_bar_actual_rate.png"))
      tryCatch(insertImage(wb, "Prediction_Means", "plot1_bar_actual_rate.png", startRow = 20, startCol = 1, width = 6, height = 4), error = function(e) {})
    if (file.exists("plot2_calibration_curve.png"))
      tryCatch(insertImage(wb, "Prediction_Means", "plot2_calibration_curve.png", startRow = 34, startCol = 1, width = 6, height = 4), error = function(e) {})
    if (file.exists("plot_overlay_with_legend_below.png"))
      tryCatch(insertImage(wb, "Prediction_Means", "plot_overlay_with_legend_below.png", startRow = 48, startCol = 1, width = 6, height = 4), error = function(e) {})
    
    addWorksheet(wb, "Misclassified")
    if (exists("misclassified_sorted")) suppressWarnings(writeData(wb, "Misclassified", misclassified_sorted))
  }
  
  saveWorkbook(wb, "Rdata_predictions.xlsx", overwrite = TRUE)
  
  # ------------------------- Debug lines ------------------------------
  if (length(best_thresholds) == 1L) {
    cat(sprintf("DEBUG >>> Final best_threshold (scalar): %s\n", if (is.finite(best_thresholds)) sprintf("%.3f", as.numeric(best_thresholds)) else "NA"))
  } else {
    cat("DEBUG >>> Final best_thresholds (per-class): ",
        paste0(ifelse(is.finite(best_thresholds), sprintf("%.3f", best_thresholds), "NA"), collapse = ", "), "\n", sep = "")
  }
  cat("DEBUG >>> Final accuracy:", round(accuracy, 5), "\n")
  
  # ------------------------- Return (back-compat + new) ---------------
  return(list(
    best_threshold   = best_threshold,       # scalar for binary, NA for multiclass
    best_thresholds  = best_thresholds,      # vector (len 1 for binary; K for multiclass - NA)
    accuracy         = accuracy,
    accuracy_percent = accuracy * 100,
    metrics          = metrics,
    misclassified    = if (exists("misclassified")) misclassified else data.frame()
  ))
}
