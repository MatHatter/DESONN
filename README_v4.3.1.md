# 🧠 DESONN Neural Network v3.0

DESONN (Dynamic Evolving Self-Organizing Neural Network) is a custom R-based neural network framework designed to support both single-layer and multi-layer architectures with dropout, bias handling, activation functions, optimizer logic, and adaptive thresholding.

# DESONN

## Ensemble labeling convention
- **E = 0** → single-run (no ensembles); stored at `ensembles$main_ensemble[[1]]` (R is 1-based).
- **E = 1** → main ensemble run.
- **E ≥ 2** → temp / prune-add ensemble passes.

**Notes**
- “E” is a label, not a list index.
- Filenames/metadata may show **E=0** (single-run) even though the container slot is `[[1]]`.


## 🚀 Version 3.0 Highlights

- 🔧 Fixed regularization logic for both **SL (single-layer)** and **ML (multi-layer)** modes:
  - Now applies `L1`, `L2`, and `L1_L2` properly to **weights** and **biases**
- 🧠 Documented that `adam` optimizer is fully routed through centralized logic in `optimizers.R` via `apply_optimizer_update()`
- 📌 Other optimizers (`SGD`, `RMSProp`, etc.) are still hardcoded in `DESONN.R` and will be moved in future versions
- ➕ Added **new custom activation functions**: `bent_relu`, `bent_sigmoid`
- 🔁 `predict()` now uses a completely separate forward-only path (no backprop logic inside)
- ✅ Accuracy: **93.7%**

## 🔍 Features

- Multi-layer and single-layer neural network support with dropout and dynamic activation chaining
- Full weight and bias updates with optimizer hooks (currently `adam` only uses `optimizers.R`)
- Threshold tuning based on F1 score using validation set
- Precision, recall, F1 evaluation built-in
- Weight imbalance correction via custom `pos_weight` / `neg_weight` logic

## 📊 Metrics Evaluation

- Accuracy, precision, recall, and F1 are active by default
- ⚠️ Additional metrics (quantization error, topographic error, MSE, MAE, DB score, diversity, etc.) are implemented in `calculate_performance()` and `calculate_relevance()` but **currently not used in training loop**

## 🧪 Experimental Logic

- Ensemble architecture is present but **not recently tested or tuned**
- Single-layer neural network (`ML_NN = FALSE`) is supported but **needs further validation**

## 📂 Output

- `Rdata_predictions.xlsx`: Full predictions on validation/test
- `misclassified_cases.xlsx`: Subset of incorrect predictions for error inspection

## 🧠 Functions

- `learn()`, `predict()` (forward-only), `apply_optimizer_update()`
- `tune_threshold()`: Auto-selects best threshold for F1
- `evaluate_classification_metrics()`: Computes F1, precision, recall
- `calculate_performance()` / `calculate_relevance()` (not active)

---

*Maintained and developed by Mathew Fok.*
