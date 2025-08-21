EvaluatePredictionsReport <- function(X_validation, y_validation, probs,
                                      predicted_outputAndTime,
                                      threshold_function,
                                      best_val_probs, best_val_labels,
                                      verbose) {
  
  # === Core: get probs & labels ===
  if (train) {
    probs  <- best_val_probs
    labels <- best_val_labels
  }
  probs  <- as.numeric(probs)
  labels <- as.numeric(labels)
  
  # === Threshold tuning ===
  threshold_result <- do.call(threshold_function, list(probs, labels))
  best_threshold   <- threshold_result$best_threshold
  if (verbose) {
    cat("Best threshold (accuracy-optimized):", best_threshold, "\n")
  }
  
  # === Binary prediction using best threshold ===
  predict_with_threshold <- function(probs, threshold) {
    ifelse(probs >= threshold, 1, 0)
  }
  binary_preds <- predict_with_threshold(probs, best_threshold)
  
  # === Accuracy ===
  labels_flat <- as.vector(labels)
  calculate_accuracy <- function(predictions, actual_labels) {
    correct_predictions <- sum(predictions == actual_labels)
    accuracy <- (correct_predictions / length(actual_labels)) * 100
    return(accuracy)
  }
  accuracy <- calculate_accuracy(binary_preds, labels_flat)
  cat("DEBUG >>> Accuracy:", accuracy, "\n")
  
  # === Precision / Recall / F1 ===
  metrics <- evaluate_classification_metrics(binary_preds, labels_flat)
  cat("DEBUG >>> Metrics:\n")
  print(metrics)
  
  # === Misclassified samples ===
  wrong <- which(binary_preds != labels_flat)
  misclassified <- data.frame(
    predicted_prob  = probs[wrong],
    predicted_label = binary_preds[wrong],
    actual_label    = labels_flat[wrong],
    as.data.frame(X_validation)[wrong, , drop = FALSE]
  )
  cat("DEBUG >>> Misclassified samples (first 10 rows):\n")
  print(head(misclassified, 10))
  
  # === Debug checkpoint ===
  cat("DEBUG >>> Final best_threshold before return:", best_threshold, "\n")
  
  # === Return results ===
  return(list(
    best_threshold = best_threshold,
    accuracy       = accuracy,
    metrics        = metrics,
    misclassified  = misclassified
  ))
}
