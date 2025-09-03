
# 🧠 DESONN Neural Network v2.9

DESONN (Dynamic Evolving Self-Organizing Neural Network) is a custom R-based neural network framework designed to support both single-layer and multi-layer architectures with dropout, bias handling, activation functions, optimizer logic, and adaptive thresholding.

## 🚀 Version 2.9 Highlights

- 🔼 Accuracy improved to **93.7%**
- ✅ Class weights manually tuned: `pos_weight = 2`, `neg_weight = 1`
- 🎯 Threshold auto-tuning added for optimal F1 score
- ⚙️ `tune_threshold()` and `evaluate_classification_metrics()` implemented
- 🧠 Support for `learn()` and `predict()` with dropout, bias broadcasting, and activation logic
- ⏱️ Learning rate scheduler function added (but not yet active)

## 🔍 Features

- Multi-layer neural network with dropout and activation chaining
- Weight and bias update with optimizer support (`adam`, `rmsprop`, etc.)
- Threshold tuning based on F1 score using validation data
- Precision, recall, F1 metrics auto-calculated
- Full prediction export with misclassified cases highlighted
- Custom loss logic and backpropagation

## 📊 Metrics Evaluation

- Accuracy, precision, recall, and F1 included by default
- ⚠️ Additional classification and performance metrics (e.g., quantization error, topographic error, Davies–Bouldin score, MSE, MAE, diversity, serendipity) are implemented in helper functions (`calculate_performance()`, `calculate_relevance()`), but currently **not activated** in the main loop

## 🧪 Experimental Logic

- Ensemble support is included but **has not been recently tested or tuned**
- Single-layer NN (`ML_NN = FALSE`) supported but **needs further validation**

## 📂 Output

- Excel exports:
  - `Rdata_predictions.xlsx`: Full predictions
  - `misclassified_cases.xlsx`: Only incorrect predictions

## 🧠 Functions

- `learn()`, `predict()`
- `tune_threshold()`: Finds best F1 threshold from 0.05 to 0.95
- `evaluate_classification_metrics()`: Calculates precision, recall, F1
- `calculate_performance()`, `calculate_relevance()`: Advanced metric tracking (currently dormant)

---

*Maintained and developed by Mathew Fok.*
