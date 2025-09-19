# ============================================
# Heart Failure (Binary) — Keras vs DESONN parity
# ============================================
# - L1 regularization (custom scale factor)
# - 200 epochs
# - Mirrors your original structure, minimal edits
# ============================================

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras import regularizers

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, average_precision_score
)

# ---------- Reproducibility ----------
SEED = 1
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------- Config ----------
USE_TIME_SPLIT = True        # True -> chronological 70/15/15; False -> stratified random 70/15/15
DATA_PATH = r"C:/Users/wfky1/Downloads/heart_failure_clinical_records.csv"

LR = 0.125
L1_LAMBDA = 0.00028
EPOCHS = 200                  # <-- 200 now
BATCH_SIZE = 16
BN_MOMENTUM = 0.9
BN_EPS = 1e-6
BN_BETA_INIT = tf.keras.initializers.Constant(0.6)
BN_GAMMA_INIT = tf.keras.initializers.Constant(0.6)

# Custom initializer scale (instead of default He scale=2.0)
CUSTOM_SCALE = 1.04349
KERNEL_INIT = VarianceScaling(
    scale=CUSTOM_SCALE, mode='fan_in', distribution='truncated_normal', seed=SEED
)

# ---------- Load & columns ----------
df = pd.read_csv(DATA_PATH).dropna()

feature_cols = [
    "age","anaemia","creatinine_phosphokinase","diabetes","ejection_fraction",
    "high_blood_pressure","platelets","serum_creatinine","serum_sodium",
    "sex","smoking","time"
]
target_col = "DEATH_EVENT"

X_all = df[feature_cols].values
y_all = df[target_col].values.astype(int)

# ---------- Split ----------
if USE_TIME_SPLIT:
    n = len(X_all)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val,   y_val   = X_all[n_train:n_train+n_val], y_all[n_train:n_train+n_val]
    X_test,  y_test  = X_all[n_train+n_val:], y_all[n_train+n_val:]
    split_note = "[SPLIT chrono]"
else:
    X_tr, X_hold, y_tr, y_hold = train_test_split(
        X_all, y_all, test_size=0.30, stratify=y_all, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold, y_hold, test_size=0.50, stratify=y_hold, random_state=SEED
    )
    X_train, y_train = X_tr, y_tr
    split_note = "[SPLIT stratified random]"

print(f"{split_note} train={len(X_train)} val={len(X_val)} test={len(X_test)}")

# ---------- Scale ----------
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ---------- Build model ----------
model = Sequential([
    Dense(64, activation="relu",
          kernel_initializer=KERNEL_INIT,
          kernel_regularizer=regularizers.l1(L1_LAMBDA),
          input_shape=(X_train.shape[1],)),
    BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS,
                       beta_initializer=BN_BETA_INIT,
                       gamma_initializer=BN_GAMMA_INIT),
    Dropout(0.10),

    Dense(32, activation="relu",
          kernel_initializer=KERNEL_INIT,
          kernel_regularizer=regularizers.l1(L1_LAMBDA)),
    BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS,
                       beta_initializer=BN_BETA_INIT,
                       gamma_initializer=BN_GAMMA_INIT),
    Dropout(0.00),

    Dense(1, activation="sigmoid",
          kernel_initializer=KERNEL_INIT,
          kernel_regularizer=regularizers.l1(L1_LAMBDA))
])

model.compile(optimizer=Adagrad(learning_rate=LR),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# ---------- Train ----------
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    verbose=2
)

# ---------- Helper: print metrics ----------
def eval_block(name, y_true, y_probs):
    y_pred = (y_probs > 0.5).astype(int)
    print(f"\n=== {name} ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=3))
    auc_roc = roc_auc_score(y_true, y_probs)
    auprc   = average_precision_score(y_true, y_probs)
    print(f"AUC (ROC): {auc_roc:.3f}")
    print(f"AUPRC:     {auprc:.3f}")

# ---------- Evaluate ----------
y_train_probs = model.predict(X_train).flatten()
y_val_probs   = model.predict(X_val).flatten()
y_test_probs  = model.predict(X_test).flatten()

eval_block("TRAIN", y_train, y_train_probs)
eval_block("VALIDATION", y_val, y_val_probs)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n[TEST] Keras evaluate -> loss={test_loss:.4f}  acc={test_acc:.4f}")
eval_block("TEST", y_test, y_test_probs)

# ---------- ROC (Validation) ----------
fpr, tpr, _ = roc_curve(y_val, y_val_probs)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc_score(y_val, y_val_probs):.3f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Validation)')
plt.legend(); plt.grid(); plt.show()
