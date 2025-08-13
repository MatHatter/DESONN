## ===== Shared plot filename helper =====
# Builds a filename prefixer for a specific context
# utils_plots.R

make_fname_prefix <- function(do_ensemble,
                              num_networks = NULL,
                              total_models = NULL,
                              ensemble_number,
                              model_index) {
  if (is.null(total_models)) total_models <- if (!is.null(num_networks)) num_networks else get0("num_networks", ifnotfound = 1L)
  ens <- as.integer(ensemble_number)
  mod <- as.integer(model_index)
  tot <- as.integer(if (length(total_models)) total_models else 1L)
  if (isTRUE(do_ensemble)) {
    return(function(base_name) sprintf("DESONN_%d_SONN_%d_%s", ens, mod, base_name))   # C/D
  }
  if (!is.na(tot) && tot > 1L) {
    return(function(base_name) sprintf("SONN_%d-%d_%s", mod, tot, base_name))          # B
    # if you prefer "SONN_<mod>_-_<tot>_<base>": sprintf("SONN_%d_-_%d_%s", mod, tot, base_name)
  }
  function(base_name) sprintf("SONN_%d_%s", mod, base_name)                            # A
}















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


