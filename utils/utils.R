tune_threshold <- function(predicted_output, labels) {
  thresholds <- seq(0.05, 0.95, by = 0.01)
  
  f1_scores <- sapply(thresholds, function(t) {
    preds <- ifelse(predicted_output >= t, 1, 0)
    TP <- sum(preds == 1 & labels == 1)
    FP <- sum(preds == 1 & labels == 0)
    FN <- sum(preds == 0 & labels == 1)
    precision <- TP / (TP + FP + 1e-8)
    recall <- TP / (TP + FN + 1e-8)
    f1 <- 2 * precision * recall / (precision + recall + 1e-8)
    return(f1)
  })
  
  best_threshold <- thresholds[which.max(f1_scores)]
  binary_preds <- ifelse(predicted_output >= best_threshold, 1, 0)
  
  return(list(
    best_threshold = best_threshold,
    binary_preds = binary_preds,
    f1_scores = f1_scores
  ))
}

tune_threshold_accuracy <- function(predicted_output, labels) {
  thresholds <- seq(0.05, 0.95, by = 0.01)
  
  accuracies <- sapply(thresholds, function(t) {
    preds <- ifelse(predicted_output >= t, 1, 0)
    sum(preds == labels) / length(labels)
  })
  
  best_threshold <- thresholds[which.max(accuracies)]
  binary_preds <- ifelse(predicted_output >= best_threshold, 1, 0)
  
  return(list(
    best_threshold = best_threshold,
    binary_preds = binary_preds,
    accuracy_scores = accuracies
  ))
}


evaluate_classification_metrics <- function(preds, labels) {
  labels <- as.vector(labels)
  preds <- as.vector(preds)
  
  TP <- sum(preds == 1 & labels == 1)
  FP <- sum(preds == 1 & labels == 0)
  FN <- sum(preds == 0 & labels == 1)
  
  precision <- TP / (TP + FP + 1e-8)
  recall <- TP / (TP + FN + 1e-8)
  F1 <- 2 * precision * recall / (precision + recall + 1e-8)
  
  return(list(precision = precision, recall = recall, F1 = F1))
}


