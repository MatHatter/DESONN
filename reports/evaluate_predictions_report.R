# ================================================================
# evaluate_predictions_report.R  (FULL — plots folder + tuned/default toggle)
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
    all_best_val_probs, all_best_val_labels,
    verbose = verbose,
    # NEW:
    accuracy_mode = c("tuned", "default", "both"),
    tuned_threshold_override = NULL
) {
  accuracy_mode <- match.arg(accuracy_mode)
  
  # ------------------------- Setup: plots dir -------------------------
  plot_dir <- file.path(getwd(), "EvaluatePredictionsReportPlots")
  if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
  
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
    tryCatch({
      png(file.path(plot_dir, "pred_vs_error_scatter.png"), width = 800, height = 600)
      plot(pred_vec[seq_len(max_points)], err_vec[seq_len(max_points)],
           main = "Prediction vs. Error", xlab = "Prediction", ylab = "Error",
           col = "steelblue", pch = 16)
      abline(h = 0, col = "gray", lty = 2)
      dev.off()
    }, error = function(e) message("⚠️ Pred-vs-Error plot failed: ", e$message))
    
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
    probs_use  <- all_best_val_probs
    labels_use <- all_best_val_labels
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
  
  # ========================= REGRESSION BRANCH =========================
  if (tolower(CLASSIFICATION_MODE) %in% c("regression", "reg")) {
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
    
    # 1) Scatter y vs yhat
    tryCatch({
      df_sc <- data.frame(y = y, yhat = yhat)
      p_sc <- ggplot(df_sc, aes(x = y, y = yhat)) +
        geom_point(alpha = 0.6) +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
        labs(title = "Regression: y (actual) vs ŷ (predicted)",
             x = "Actual", y = "Predicted") +
        theme_minimal()
      ggsave(file.path(plot_dir, "reg_scatter_y_vs_yhat.png"), p_sc, width = 6, height = 4, dpi = 300)
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
      ggsave(file.path(plot_dir, "reg_residuals_vs_fitted.png"), p_rf, width = 6, height = 4, dpi = 300)
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
      ggsave(file.path(plot_dir, "reg_residuals_hist.png"), p_rh, width = 6, height = 4, dpi = 300)
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
    if (file.exists(file.path(plot_dir, "reg_scatter_y_vs_yhat.png"))) {
      tryCatch(insertImage(wb, "Metrics_Summary", file.path(plot_dir, "reg_scatter_y_vs_yhat.png"), startRow = 20, startCol = 1, width = 6, height = 4), error = function(e) {})
    }
    if (file.exists(file.path(plot_dir, "reg_residuals_vs_fitted.png"))) {
      tryCatch(insertImage(wb, "Metrics_Summary", file.path(plot_dir, "reg_residuals_vs_fitted.png"), startRow = 35, startCol = 1, width = 6, height = 4), error = function(e) {})
    }
    if (file.exists(file.path(plot_dir, "reg_residuals_hist.png"))) {
      tryCatch(insertImage(wb, "Metrics_Summary", file.path(plot_dir, "reg_residuals_hist.png"), startRow = 50, startCol = 1, width = 6, height = 4), error = function(e) {})
    }
    saveWorkbook(wb, "Rdata_predictions.xlsx", overwrite = TRUE)
    
    return(list(
      best_threshold   = NA_real_,
      best_thresholds  = NA_real_,
      accuracy         = NA_real_,
      accuracy_percent = NA_real_,
      metrics          = list(RMSE = RMSE, MAE = MAE, MAPE = MAPE, R2 = R2, Correlation = Corr),
      y_pred_class_default = NULL,
      y_pred_class_tuned   = NULL,
      misclassified    = data.frame()
    ))
  }
  # ======================= END REGRESSION BRANCH =======================
  
  # Infer binary vs multiclass using helpers from utils
  inf <- infer_is_binary(labels_mat, probs_mat)
  is_binary <- isTRUE(inf$is_binary)
  K <- max(1L, ncol(labels_mat), ncol(probs_mat))
  
  # ------------------------- DEBUG DIAGNOSTICS -------------------------
  if(verbose){
    cat("\n[DBG] --- threshold tuning inputs ---\n")
    dbg_shape <- function(x) {
      nr <- if (!is.null(dim(x))) nrow(x) else length(x)
      nc <- if (!is.null(dim(x))) ncol(x) else 1L
      tp <- paste(class(x), collapse = "/")
      c(nrow = nr, ncol = nc, type = tp)
    }
    dbg_count_nonfinite <- function(x) {
      if (is.null(dim(x))) x <- matrix(x, ncol = 1L)
      nf  <- sum(!is.finite(x)); na_ <- sum(is.na(x))
      nan <- sum(is.nan(x)); infv <- sum(is.infinite(x))
      setNames(c(nf, na_, nan, infv), c("nonfinite", "NA", "NaN", "Inf"))
    }
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
  
  # ------------------------- Helpers -------------------------
  labels_to_binary_vec_safe <- function(L) labels_to_binary_vec(L)
  preds_to_pos_prob_safe    <- function(P) preds_to_pos_prob(P)
  
  # --- Storage for dual outputs ---
  results <- list(
    best_threshold         = NA_real_,
    best_thresholds        = NA_real_,
    accuracy               = NA_real_,  # legacy mirror of selected mode
    accuracy_percent       = NA_real_,
    metrics                = NULL,
    y_pred_class_default   = NULL,
    y_pred_class_tuned     = NULL,
    accuracy_default       = NA_real_,
    accuracy_tuned         = NA_real_,
    precision_default      = NA_real_,
    recall_default         = NA_real_,
    f1_default             = NA_real_,
    precision_tuned        = NA_real_,
    recall_tuned           = NA_real_,
    f1_tuned               = NA_real_,
    misclassified          = data.frame()
  )
  
  # =============================== BINARY ===============================
  if (is_binary) {
    labels_flat <- labels_to_binary_vec_safe(labels_mat)
    p_pos       <- preds_to_pos_prob_safe(probs_mat)
    if (is.null(labels_flat) || is.null(p_pos)) {
      stop("Binary inference succeeded, but could not derive binary labels/probabilities.")
    }
    
    # ----- DEFAULT (0.5) PATH -----
    binary_preds_default <- as.integer(p_pos >= 0.5)
    acc_default <- accuracy(
      SONN = NULL, Rdata = NULL,
      labels = matrix(labels_flat, ncol = 1L),
      CLASSIFICATION_MODE = CLASSIFICATION_MODE,
      predicted_output = matrix(binary_preds_default, ncol = 1L),
      verbose = FALSE
    )
    metrics_default <- safe_eval_metrics(binary_preds_default, labels_flat, verbose = FALSE)
    
    # ----- TUNED PATH -----
    thr_grid <- sanitize_grid(seq(0.05, 0.95, by = 0.01))
    if (is.numeric(tuned_threshold_override) && is.finite(tuned_threshold_override)) {
      best_thresholds <- tuned_threshold_override
      if (verbose) cat(sprintf("[TUNE] Using tuned_threshold_override=%.3f (no re-tune)\n", tuned_threshold_override))
    } else {
      thr_res <- tryCatch(
        tune_threshold_accuracy(
          predicted_output = matrix(p_pos, ncol = 1L),
          labels           = matrix(labels_flat, ncol = 1L),
          metric           = "accuracy",
          threshold_grid   = thr_grid,
          verbose          = FALSE
        ),
        error = function(e) {
          message("tune_threshold_accuracy failed: ", e$message, " — using 0.5.")
          list(thresholds = 0.5, y_pred_class = NULL)
        }
      )
      best_thresholds <- thr_res$thresholds
    }
    best_threshold <- as.numeric(best_thresholds)
    
    binary_preds_tuned <- as.integer(p_pos >= best_threshold)
    acc_tuned <- accuracy(
      SONN = NULL, Rdata = NULL,
      labels = matrix(labels_flat, ncol = 1L),
      CLASSIFICATION_MODE = CLASSIFICATION_MODE,
      predicted_output = matrix(binary_preds_tuned, ncol = 1L),
      verbose = FALSE
    )
    metrics_tuned <- safe_eval_metrics(binary_preds_tuned, labels_flat, verbose = FALSE)
    
    # Which modes to build artifacts for
    use_tuned   <- accuracy_mode %in% c("tuned", "both")
    use_default <- accuracy_mode %in% c("default", "both")
    
    # ---------- Artifact builder per mode ----------
    build_binary_artifacts <- function(mode_label, bin_preds, threshold_used) {
      suffix <- if (mode_label == "tuned") "_tuned" else "_default"
      
      # Confusion counts
      TP <- sum(bin_preds == 1 & labels_flat == 1)
      TN <- sum(bin_preds == 0 & labels_flat == 0)
      FP <- sum(bin_preds == 1 & labels_flat == 0)
      FN <- sum(bin_preds == 0 & labels_flat == 1)
      
      conf_matrix_df <- data.frame(
        Actual    = c("0","0","1","1"),
        Predicted = c("0","1","0","1"),
        Count     = c(TN, FP, FN, TP)
      )
      
      # Heatmap
      heatmap_path <- file.path(plot_dir, paste0("confusion_matrix_heatmap", suffix, ".png"))
      tryCatch({
        plot_conf_matrix <- ggplot(conf_matrix_df, aes(x = Predicted, y = Actual, fill = Count)) +
          geom_tile(color = "white") +
          geom_text(aes(label = Count), size = 6, fontface = "bold") +
          scale_fill_gradient(low = "white", high = "red") +
          labs(title = paste("Confusion Matrix Heatmap", toupper(mode_label))) +
          theme_minimal() +
          theme(plot.title = element_text(hjust = 0.5, face = "bold"))
        ggsave(heatmap_path, plot_conf_matrix, width = 5, height = 4, dpi = 300)
        while (!is.null(dev.list())) dev.off()
        if (verbose) print(plot_conf_matrix)
      }, error = function(e) message("❌ Failed to generate confusion matrix heatmap: ", e$message))
      
      # Combined DF
      Rdata_df <- as.data.frame(X_validation)
      combined_df <- cbind(
        Rdata_df,
        label = labels_flat,
        pred  = bin_preds,
        prob  = as.numeric(p_pos),
        threshold_used = threshold_used
      )

      
      # Probability separation / calibration
      Rdata_predictions_cal <- cbind(as.data.frame(X_validation), Label = labels_flat) %>%
        transmute(prob = as.numeric(p_pos),
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
      
      # Plot 1: Observed rate by decile
      plot1_path <- file.path(plot_dir, paste0("plot1_bar_actual_rate", suffix, ".png"))
      tryCatch({
        plot1 <- ggplot(Rdata_predictions_cal, aes(x = prob_bin, y = actual_death_rate)) +
          geom_col(fill = "steelblue") +
          labs(title = paste("Observed Rate by Risk Bin (", mode_label, ")", sep = ""),
               x = "Predicted Risk Decile (1=low,10=high)",
               y = "Observed Positive Rate") +
          theme_minimal() +
          theme(plot.title = element_text(face = "bold", hjust = 0.5))
        ggsave(plot1_path, plot1, width = 6, height = 4, dpi = 300)
        while (!is.null(dev.list())) dev.off()
        if (verbose) print(plot1)
      }, error = function(e) message("❌ Failed to generate plot1: ", e$message))
      
      # Plot 2: Calibration curve
      plot2_path <- file.path(plot_dir, paste0("plot2_calibration_curve", suffix, ".png"))
      tryCatch({
        plot2 <- ggplot(Rdata_predictions_cal, aes(x = bin_mid, y = actual_death_rate)) +
          geom_line(size = 1.2) +
          geom_point(size = 3) +
          geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
          labs(title = paste("Calibration Curve (", mode_label, ")", sep = ""),
               x = "Avg Predicted Probability",
               y = "Observed Rate") +
          theme_minimal() +
          theme(plot.title = element_text(face = "bold", hjust = 0.5))
        ggsave(plot2_path, plot2, width = 6, height = 4, dpi = 300)
        while (!is.null(dev.list())) dev.off()
        if (verbose) print(plot2)
      }, error = function(e) message("❌ Failed to generate plot2: ", e$message))
      
      # Overlay: Observed vs Predicted
      overlay_path <- file.path(plot_dir, paste0("plot_overlay_with_legend_below", suffix, ".png"))
      tryCatch({
        plot_overlay <- ggplot(Rdata_predictions_cal, aes(x = prob_bin)) +
          geom_col(aes(y = actual_death_rate, fill = "Observed Rate")) +
          geom_point(aes(y = bin_mid, color = "Avg Predicted Prob"),
                     size = 3, shape = 21, stroke = 1.2, fill = "white") +
          scale_fill_manual(values = c("Observed Rate" = "steelblue")) +
          scale_color_manual(values = c("Avg Predicted Prob" = "black")) +
          labs(title = paste("Overlay: Observed vs Predicted (", mode_label, ")", sep = ""),
               x = "Predicted Risk Decile", y = "Rate", fill = NULL, color = NULL) +
          theme_minimal() +
          theme(legend.position = "bottom",
                plot.title = element_text(face = "bold", hjust = 0.5))
        ggsave(overlay_path, plot_overlay, width = 6, height = 4, dpi = 300)
        while (!is.null(dev.list())) dev.off()
        if (verbose) print(plot_overlay)
      }, error = function(e) message("❌ Failed to generate overlay plot: ", e$message))
      
      # Means commentary
      prediction_means <- data.frame(
        Label = c(0, 1),
        MeanProbability = c(
          mean(as.numeric(p_pos[labels_flat == 0]), na.rm = TRUE),
          mean(as.numeric(p_pos[labels_flat == 1]), na.rm = TRUE)
        ),
        StdDev = c(
          sd(as.numeric(p_pos[labels_flat == 0]), na.rm = TRUE),
          sd(as.numeric(p_pos[labels_flat == 1]), na.rm = TRUE)
        ),
        Count = c(sum(labels_flat == 0), sum(labels_flat == 1))
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
      
      # ROC & PR: same probabilities; save per mode for clarity
      roc_path <- file.path(plot_dir, paste0("roc_curve", suffix, ".png"))
      pr_path  <- file.path(plot_dir, paste0("pr_curve",  suffix, ".png"))
      tryCatch({
        labels_numeric <- suppressWarnings(as.numeric(labels_flat))
        probs_numeric  <- suppressWarnings(as.numeric(p_pos))
        if (length(unique(labels_numeric)) >= 2 && length(labels_numeric) == length(probs_numeric)) {
          roc_obj <- pROC::roc(response = labels_numeric, predictor = probs_numeric, quiet = TRUE)
          auc_value <- pROC::auc(roc_obj)
          auc_val_text <- paste("AUC =", round(as.numeric(auc_value), 4))
          roc_curve_plot <- ggplotify::as.ggplot(~{
            plot(roc_obj, col = "#1f77b4", lwd = 2, main = paste("ROC Curve", toupper(mode_label)))
            abline(a = 0, b = 1, lty = 2, col = "gray")
            text(0.6, 0.2, labels = auc_val_text, cex = 1.2)
          })
          ggsave(roc_path, plot = roc_curve_plot, width = 6, height = 4, dpi = 300)
          while (!is.null(dev.list())) dev.off()
          if (verbose) print(roc_curve_plot)
        }
      }, error = function(e) message("⚠️ ROC block failed: ", e$message))
      tryCatch({
        labels_numeric <- suppressWarnings(as.numeric(labels_flat))
        probs_numeric  <- suppressWarnings(as.numeric(p_pos))
        if (length(unique(labels_numeric)) >= 2 && length(labels_numeric) == length(probs_numeric)) {
          pr_obj <- PRROC::pr.curve(
            scores.class0 = probs_numeric[labels_numeric == 1],
            scores.class1 = probs_numeric[labels_numeric == 0],
            curve = TRUE
          )
          pr_curve_plot <- ggplotify::as.ggplot(~{
            plot(pr_obj, main = paste("Precision-Recall Curve", toupper(mode_label)), col = "#d62728", lwd = 2)
          })
          ggsave(pr_path, plot = pr_curve_plot, width = 6, height = 4, dpi = 300)
          while (!is.null(dev.list())) dev.off()
          if (verbose) print(pr_curve_plot)
        }
      }, error = function(e) message("⚠️ PR block failed: ", e$message))
      
      # Misclassified (sorted)
      # Misclassified (sorted)
      wrong <- which(bin_preds != labels_flat)
      misclassified_sorted <- if (length(wrong)) {
        df <- cbind(
          predicted_prob = as.numeric(p_pos[wrong]),
          pred           = bin_preds[wrong],         # in tuned mode this is tuned; in default mode it's default
          actual_label   = labels_flat[wrong],
          as.data.frame(X_validation)[wrong, , drop = FALSE]
        )
        as.data.frame(df) %>%
          mutate(
            error = abs(predicted_prob - actual_label),
            Type  = ifelse(pred == 1 & actual_label == 0, "False Positive", "False Negative")
          ) %>%
          arrange(desc(error))
      } else data.frame()
      
      
      
      # Metrics summary table (per-mode)
      total    <- length(labels_flat)
      support0 <- sum(labels_flat == 0)
      support1 <- sum(labels_flat == 1)
      precision1 <- if ((TP + FP) > 0) TP / (TP + FP) else 0
      recall1    <- if ((TP + FN) > 0) TP / (TP + FN) else 0
      f1_1       <- if ((precision1 + recall1) > 0) 2 * precision1 * recall1 / (precision1 + recall1) else 0
      precision0 <- if ((TN + FN) > 0) TN / (TN + FN) else 0
      recall0    <- if ((TN + FP) > 0) TN / (TN + FP) else 0
      f1_0       <- if ((precision0 + recall0) > 0) 2 * precision0 * recall0 / (precision0 + recall0) else 0
      
      macro_precision <- mean(c(precision0, precision1))
      macro_recall    <- mean(c(recall0,    recall1))
      macro_f1        <- mean(c(f1_0,       f1_1))
      weighted_precision <- (support0 * precision0 + support1 * precision1) / max(total, 1)
      weighted_recall    <- (support0 * recall0 + support1 * recall1) / max(total, 1)
      weighted_f1        <- (support0 * f1_0 + support1 * f1_1) / max(total, 1)
      
      metrics_summary <- data.frame(
        Class      = c("0", "1", "macro avg", "weighted avg"),
        Precision  = c(precision0, precision1, macro_precision, weighted_precision),
        Recall     = c(recall0,    recall1,    macro_recall,    weighted_recall),
        F1_Score   = c(f1_0,       f1_1,       macro_f1,        weighted_f1),
        Support    = c(support0,   support1,   total,           total),
        Accuracy   = rep((TP + TN) / max(1, TP + TN + FP + FN), 4),
        TP         = c(TN, TP, NA_real_, NA_real_),
        TN         = c(TN, TP, NA_real_, NA_real_),
        FP         = rep(FP, 4),
        FN         = rep(FN, 4)
      )
      
      # Save metrics table PNG per mode
      metrics_png_path <- file.path(plot_dir, paste0("metrics_summary", suffix, ".png"))
      tryCatch({
        png(metrics_png_path, width = 1000, height = 400)
        grid.table(metrics_summary)
        dev.off()
      }, error = function(e) message("❌ Failed to save metrics_summary PNG: ", e$message))
      
      list(
        suffix = suffix,
        threshold_used = threshold_used,
        conf_matrix_df = conf_matrix_df,
        heatmap_path = heatmap_path,
        combined_df  = combined_df,
        prediction_means = prediction_means,
        commentary_df_means = commentary_df_means,
        misclassified_sorted = misclassified_sorted,
        metrics_summary = metrics_summary,
        plot1_path = plot1_path,
        plot2_path = plot2_path,
        overlay_path = overlay_path,
        roc_path = roc_path,
        pr_path  = pr_path,
        metrics_png_path = metrics_png_path
      )
    } # end build_binary_artifacts
    
    # Build selected artifacts
    artifacts <- list()
    if (use_default) artifacts$default <- build_binary_artifacts("default", binary_preds_default, 0.5)
    if (use_tuned)   artifacts$tuned   <- build_binary_artifacts("tuned",   binary_preds_tuned,   best_threshold)
    
    # ---------- Write Excel with one or both modes ----------
    wb <- createWorkbook()
    
    for (mode_name in names(artifacts)) {
      art <- artifacts[[mode_name]]
      sheet_prefix <- if (mode_name == "tuned") "Tuned" else "Default"
      comb_sheet <- paste0(sheet_prefix, "_Combined")
      metr_sheet <- paste0(sheet_prefix, "_Metrics")
      pred_sheet <- paste0(sheet_prefix, "_Prediction_Means")
      misc_sheet <- paste0(sheet_prefix, "_Misclassified")
      
      addWorksheet(wb, comb_sheet)
      suppressWarnings(writeData(wb, comb_sheet, art$combined_df))
      
      addWorksheet(wb, metr_sheet)
      suppressWarnings(writeData(wb, metr_sheet, art$metrics_summary))
      if (exists("art$conf_matrix_df")) {
        suppressWarnings(writeData(wb, metr_sheet, art$conf_matrix_df, startRow = nrow(art$metrics_summary) + 6, rowNames = FALSE))
      }
      # Insert images (heatmap + ROC/PR + metrics PNG)
      if (file.exists(art$heatmap_path)) {
        tryCatch(insertImage(wb, metr_sheet, art$heatmap_path, startRow = nrow(art$metrics_summary) + 12, startCol = 1, width = 6, height = 4),
                 error = function(e) message("⚠️ Could not insert heatmap: ", e$message))
      }
      if (file.exists(art$roc_path)) {
        tryCatch(insertImage(wb, metr_sheet, art$roc_path, startRow = nrow(art$metrics_summary) + 26, startCol = 1, width = 6, height = 4),
                 error = function(e) message("⚠️ Could not insert ROC: ", e$message))
      }
      if (file.exists(art$pr_path)) {
        tryCatch(insertImage(wb, metr_sheet, art$pr_path, startRow = nrow(art$metrics_summary) + 40, startCol = 1, width = 6, height = 4),
                 error = function(e) message("⚠️ Could not insert PR: ", e$message))
      }
      if (file.exists(art$metrics_png_path)) {
        tryCatch(insertImage(wb, metr_sheet, art$metrics_png_path, startRow = nrow(art$metrics_summary) + 54, startCol = 1, width = 6, height = 4),
                 error = function(e) message("⚠️ Could not insert metrics PNG: ", e$message))
      }
      
      addWorksheet(wb, pred_sheet)
      if (nrow(art$prediction_means) > 0) suppressWarnings(writeData(wb, pred_sheet, art$prediction_means))
      if (nrow(art$commentary_df_means) > 0) suppressWarnings(writeData(wb, pred_sheet, art$commentary_df_means, startRow = nrow(art$prediction_means) + 3))
      if (file.exists(art$plot1_path)) {
        tryCatch(insertImage(wb, pred_sheet, art$plot1_path, startRow = 20, startCol = 1, width = 6, height = 4), error = function(e) {})
      }
      if (file.exists(art$plot2_path)) {
        tryCatch(insertImage(wb, pred_sheet, art$plot2_path, startRow = 34, startCol = 1, width = 6, height = 4), error = function(e) {})
      }
      if (file.exists(art$overlay_path)) {
        tryCatch(insertImage(wb, pred_sheet, art$overlay_path, startRow = 48, startCol = 1, width = 6, height = 4), error = function(e) {})
      }
      
      addWorksheet(wb, misc_sheet)
      if (nrow(art$misclassified_sorted) > 0) suppressWarnings(writeData(wb, misc_sheet, art$misclassified_sorted))
    }
    
    saveWorkbook(wb, "Rdata_predictions.xlsx", overwrite = TRUE)
    
    # ---- Populate return fields ----
    results$best_threshold   <- best_threshold
    results$best_thresholds  <- best_thresholds
    results$accuracy_default <- acc_default
    results$accuracy_tuned   <- acc_tuned
    results$precision_default <- metrics_default$precision
    results$recall_default    <- metrics_default$recall
    results$f1_default        <- metrics_default$F1
    results$precision_tuned   <- metrics_tuned$precision
    results$recall_tuned      <- metrics_tuned$recall
    results$f1_tuned          <- metrics_tuned$F1
    results$y_pred_class_default <- binary_preds_default
    results$y_pred_class_tuned   <- binary_preds_tuned
    
    # Back-compat: mirror selected mode into legacy fields
    if (accuracy_mode == "tuned") {
      results$accuracy         <- acc_tuned
      results$accuracy_percent <- acc_tuned * 100
      results$metrics          <- metrics_tuned
      if (!is.null(artifacts$tuned)) {
        results$misclassified <- if (nrow(artifacts$tuned$misclassified_sorted) > 0) artifacts$tuned$misclassified_sorted else data.frame()
      }
    } else if (accuracy_mode == "default") {
      results$accuracy         <- acc_default
      results$accuracy_percent <- acc_default * 100
      results$metrics          <- metrics_default
      if (!is.null(artifacts$default)) {
        results$misclassified <- if (nrow(artifacts$default$misclassified_sorted) > 0) artifacts$default$misclassified_sorted else data.frame()
      }
    } else { # both
      results$accuracy         <- acc_tuned
      results$accuracy_percent <- acc_tuned * 100
      results$metrics          <- metrics_tuned
      if (!is.null(artifacts$tuned)) {
        results$misclassified <- if (nrow(artifacts$tuned$misclassified_sorted) > 0) artifacts$tuned$misclassified_sorted else data.frame()
      }
    }
    
    if (verbose) {
      cat(sprintf("Acc @0.50 = %.4f | Acc @tuned(%.2f) = %.4f\n",
                  acc_default, best_threshold, acc_tuned))
    }
    
    return(results)
  }
  
  # =============================== MULTICLASS ===============================
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
  
  predicted_output_for_acc <- one_hot_from_ids(pred_ids, K, N_eff)
  
  acc_mc <- accuracy(
    SONN = NULL, Rdata = NULL,
    labels = labels_mat,
    CLASSIFICATION_MODE = CLASSIFICATION_MODE,
    predicted_output = predicted_output_for_acc,
    verbose = FALSE
  )
  accuracy_val <- acc_mc
  accuracy_percent <- accuracy_val * 100
  if (verbose) message(sprintf("Accuracy (multiclass, argmax): %.6f (%.3f%%)", accuracy_val, accuracy_percent))
  
  conf_tab <- table(
    Actual    = factor(y_true_ids, levels = 1:K),
    Predicted = factor(pred_ids,   levels = 1:K)
  )
  conf_matrix_df <- as.data.frame(conf_tab); names(conf_matrix_df)[3] <- "Count"
  
  # Heatmap (multiclass)
  heatmap_path_mc <- file.path(plot_dir, "confusion_matrix_multiclass_heatmap.png")
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
    ggsave(heatmap_path_mc, plot_conf_matrix_mc, width = 6, height = 5, dpi = 300)
    while (!is.null(dev.list())) dev.off()
    if (verbose) print(plot_conf_matrix_mc)
  }, error = function(e) message("❌ Failed to generate multiclass heatmap: ", e$message))
  
  Rdata_df <- as.data.frame(X_validation)
  combined_df <- cbind(
    Rdata_df,
    label = y_true_ids,
    pred  = pred_ids
  )
  
  # Build per-class metrics
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
  
  metrics_summary_mc <- rbind(
    data.frame(
      Class     = as.character(seq_len(K)),
      Precision = Prec_k,
      Recall    = Rec_k,
      F1_Score  = F1_k,
      Support   = Support,
      Accuracy  = rep(accuracy_val, K),
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
      Accuracy  = c(accuracy_val, accuracy_val),
      TP        = c(NA_real_, NA_real_),
      TN        = c(NA_real_, NA_real_),
      FP        = c(NA_real_, NA_real_),
      FN        = c(NA_real_, NA_real_)
    )
  )
  
  metrics_png_path_mc <- file.path(plot_dir, "metrics_summary_multiclass.png")
  tryCatch({
    png(metrics_png_path_mc, width = 1000, height = 400)
    grid.table(metrics_summary_mc)
    dev.off()
  }, error = function(e) message("❌ Failed to save metrics_summary_multiclass.png: ", e$message))
  
  wb <- createWorkbook()
  addWorksheet(wb, "Combined")
  suppressWarnings(writeData(wb, "Combined", combined_df))
  addWorksheet(wb, "Metrics_Summary")
  suppressWarnings(writeData(wb, "Metrics_Summary", metrics_summary_mc))
  if (exists("conf_matrix_df")) {
    suppressWarnings(writeData(wb, "Metrics_Summary", conf_matrix_df, startRow = nrow(metrics_summary_mc) + 6, rowNames = FALSE))
  }
  if (file.exists(heatmap_path_mc)) {
    tryCatch(insertImage(wb, "Metrics_Summary", heatmap_path_mc, startRow = nrow(metrics_summary_mc) + 12, startCol = 1, width = 6, height = 4),
             error = function(e) message("⚠️ Could not insert heatmap: ", e$message))
  }
  if (file.exists(metrics_png_path_mc)) {
    tryCatch(insertImage(wb, "Metrics_Summary", metrics_png_path_mc, startRow = nrow(metrics_summary_mc) + 26, startCol = 1, width = 6, height = 4),
             error = function(e) message("⚠️ Could not insert metrics PNG: ", e$message))
  }
  saveWorkbook(wb, "Rdata_predictions.xlsx", overwrite = TRUE)
  
  return(list(
    best_threshold         = NA_real_,
    best_thresholds        = NA_real_,
    accuracy               = accuracy_val,
    accuracy_percent       = accuracy_val * 100,
    metrics                = list(precision = macro_precision, recall = macro_recall, F1 = macro_f1),
    y_pred_class_default   = NULL,
    y_pred_class_tuned     = NULL,
    accuracy_default       = NA_real_,
    accuracy_tuned         = NA_real_,
    precision_default      = NA_real_,
    recall_default         = NA_real_,
    f1_default             = NA_real_,
    precision_tuned        = NA_real_,
    recall_tuned           = NA_real_,
    f1_tuned               = NA_real_,
    misclassified          = data.frame()
  ))
}
